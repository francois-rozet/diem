#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from datasets import load_from_disk, Array3D, Features
from dawgz import job, schedule
from functools import partial
from tqdm import trange
from typing import *

from utils import *


CONFIG = {
    # Architecture
    'hid_channels': (128, 256, 384),
    'hid_blocks': (5, 5, 5),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {2: 4},
    'dropout': 0.1,
    # Training
    'laps': 16,
    'epochs': 256,
    'batch_size': 256,
    'scheduler': 'constant',
    'lr_init': 2e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.999,
}


def generate(model, dataset, rng, batch_size, sharding=None):
    def transform(batch):
        y, A = batch['y'], batch['A']
        y, A = jax.device_put((y, A), sharding)
        x = sample(model, y, A, rng.split())
        x = np.asarray(x)

        return {'x': x}

    types = {'x': Array3D(shape=(32, 32, 3), dtype='float32')}

    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=['y', 'A'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )


def train(runid: int, lap: int):
    run = wandb.init(
        project='priors-cifar-mask',
        id=runid,
        resume='allow',
        dir=PATH,
        config=CONFIG,
    )

    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # Sharding
    jax.config.update('jax_threefry_partitionable', True)

    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # RNG
    seed = hash((runpath, lap)) % 2**16
    rng = inox.random.PRNG(seed)

    # Data
    dataset = load_from_disk(PATH / 'hf/cifar-mask')
    dataset.set_format('numpy')

    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    y_eval, A_eval = testset_yA[:16]['y'], testset_yA[:16]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    # Previous
    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        y_fit, A_fit = trainset_yA[:16384]['y'], trainset_yA[:16384]['A']
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

        mu_x, sigma_x = fit_moments(
            features=32 * 32 * 3,
            rank=64,
            A=inox.Partial(measure, A_fit),
            y=flatten(y_fit),
            sigma_y=1e-3 ** 2,
            sampler='ddim',
            steps=256,
            maxiter=None,
            key=rng.split(),
        )

        del y_fit, A_fit

        previous = GaussianDenoiser(mu_x, sigma_x)

    ## Generate
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    trainset = generate(previous, trainset_yA, rng, config.batch_size, distributed)
    testset = generate(previous, testset_yA, rng, config.batch_size, distributed)

    ## Moments
    x_fit = trainset[:16384]['x']
    x_fit = flatten(x_fit)

    mu_x = jnp.mean(x_fit, axis=0)

    del x_fit

    # Model
    if lap > 0:
        model = previous
    else:
        model = make_model(key=rng.split(), **config)

    model.mu_x = mu_x
    model.train(True)

    static, params, others = model.partition(nn.Parameter)

    # Objective
    objective = DenoiserLoss()

    # Optimizer
    steps = config.epochs * len(trainset_yA) // config.batch_size
    optimizer = Adam(steps=steps, **config)
    opt_state = optimizer.init(params)

    # EMA
    ema = EMA(decay=config.ema_decay)
    avrg = params

    # Training
    avrg, params, others, opt_state = jax.device_put((avrg, params, others, opt_state), replicated)

    @jax.jit
    @jax.vmap
    def augment(x, key):
        keys = jax.random.split(key, 2)

        x = rand_flip(x, keys[0], axis=-2)
        x = rand_shake(x, keys[1], delta=1)

        return x

    @jax.jit
    def ell(params, others, x, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])

        return objective(static(params, others), x, z, t, key=keys[2])

    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)

        return loss, avrg, params, opt_state

    for epoch in (bar := trange(config.epochs, ncols=88)):
        loader = (
            trainset
            .shuffle(seed=seed + lap * config.epochs + epoch)
            .iter(batch_size=config.batch_size, drop_last_batch=True)
        )

        losses = []

        for batch in prefetch(loader):
            x = batch['x']
            x = jax.device_put(x, distributed)
            x = augment(x, rng.split(len(x)))
            x = flatten(x)

            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, key=rng.split())
            losses.append(loss)

        loss_train = np.stack(losses).mean()

        ## Validation
        loader = (
            testset
            .iter(batch_size=config.batch_size, drop_last_batch=True)
        )

        losses = []

        for batch in prefetch(loader):
            x = batch['x']
            x = jax.device_put(x, distributed)
            x = flatten(x)

            loss = ell(avrg, others, x, key=rng.split())
            losses.append(loss)

        loss_val = np.stack(losses).mean()

        bar.set_postfix(loss=loss_train, loss_val=loss_val)

        ## Eval
        if (epoch + 1) % 16 == 0:
            model = static(avrg, others)
            model.train(False)

            x = sample(model, y_eval, A_eval, rng.split())
            x = x.reshape(4, 4, 32, 32, 3)

            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'samples': wandb.Image(to_pil(x, zoom=4)),
            })
        else:
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
            })

    ## Checkpoint
    model = static(avrg, others)
    model.train(False)

    dump_module(model, runpath / f'checkpoint_{lap}.pkl')


if __name__ == '__main__':
    runid = wandb.util.generate_id()

    jobs = []

    for lap in range(CONFIG.get('laps')):
        jobs.append(
            job(
                partial(train, runid=runid, lap=lap),
                name=f'train_{lap}',
                cpus=4,
                gpus=4,
                ram='64GB',
                time='1-00:00:00',
                partition='gpu',
            )
        )

        if lap > 0:
            jobs[lap].after(jobs[lap - 1])

    schedule(
        *jobs,
        name=f'Training {runid}',
        backend='slurm',
        export='ALL',
        account='ariacpg',
        env=['export WANDB_SILENT=true'],
    )
