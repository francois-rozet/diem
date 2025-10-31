#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from datasets import Array3D, Features, concatenate_datasets, load_from_disk
from dawgz import job, schedule
from functools import partial
from tqdm import trange
from typing import *

# isort: split
from utils import *

CONFIG = {
    # Data
    'duplicate': 2,
    # Architecture
    'hid_channels': (128, 256, 384, 512),
    'hid_blocks': (3, 3, 3, 3),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {3: 4},
    'dropout': 0.1,
    # Sampling
    'sampler': 'ddpm',
    'heuristic': None,
    'sde': {'a': 1e-3, 'b': 1e2},
    'discrete': 64,
    'maxiter': 3,
    # Training
    'epochs': 64,
    'batch_size': 256,
    'scheduler': 'constant',
    'lr_init': 1e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.999,
}


def generate(model, dataset, rng, batch_size, **kwargs):
    def transform(batch):
        y, A = batch['y'], batch['A']
        x = sample(model, y, A, rng.split(), **kwargs)
        x = np.asarray(x)

        return {'x': x}

    types = {'x': Array3D(shape=(320, 320, 1), dtype='float32')}

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
        project='diem-fastmri-kspace',
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

    # SDE
    sde = VESDE(**CONFIG.get('sde'))

    # Data
    dataset = load_from_disk(PATH / 'hf/fastmri-kspace')
    dataset.set_format('numpy')

    trainset_yA = dataset['train']
    trainset_yA = concatenate_datasets([trainset_yA] * config.duplicate)
    testset_yA = dataset['val']

    y_eval, A_eval = testset_yA[:1024:256]['y'], testset_yA[:1024:256]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    # Previous
    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        y_fit, A_fit = trainset_yA[:16384:4]['y'], trainset_yA[:16384:4]['A']
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)

        mu_x, cov_x = fit_moments(
            features=320 * 320 * 1,
            rank=64,
            shard=True,
            A=inox.Partial(measure, A_fit, shard=True),
            y=flatten(y_fit),
            cov_y=1e-2**2,
            sampler='ddim',
            sde=sde,
            steps=256,
            maxiter=5,
            key=rng.split(),
        )

        del y_fit, A_fit

        previous = GaussianDenoiser(mu_x, cov_x)

    ## Generate
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    trainset = generate(
        model=previous,
        dataset=trainset_yA,
        rng=rng,
        batch_size=config.batch_size,
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )
    testset = generate(
        model=previous,
        dataset=testset_yA,
        rng=rng,
        batch_size=config.batch_size,
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )

    ## Moments
    x_fit = trainset[:16384]['x']
    x_fit = flatten(x_fit)
    x_fit = jax.device_put(x_fit, distributed)

    mu_x, cov_x = ppca(x_fit, rank=64, key=rng.split())

    del x_fit

    # Model
    if lap > 0:
        model = previous
    else:
        model = make_model(key=rng.split(), **CONFIG)

    model.mu_x = mu_x

    if config.heuristic == 'zeros':
        model.cov_x = jnp.zeros_like(mu_x)
    elif config.heuristic == 'ones':
        model.cov_x = jnp.ones_like(mu_x)
    elif config.heuristic == 'cov_t':
        model.cov_x = jnp.ones_like(mu_x) * 1e6
    elif config.heuristic == 'cov_x':
        model.cov_x = cov_x

    model.train(True)

    static, params, others = model.partition(nn.Parameter)

    # Objective
    objective = DenoiserLoss(sde=sde)

    # Optimizer
    steps = config.epochs * len(dataset) // config.batch_size
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

        x = random_flip(x, keys[0], axis=-2)
        x = random_shake(x, keys[1], delta=4)

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
        loader = trainset.shuffle(seed=seed + lap * config.epochs + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
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
        loader = testset.iter(batch_size=config.batch_size, drop_last_batch=True)

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

            x = sample(
                model=model,
                y=y_eval,
                A=A_eval,
                key=rng.split(),
                shard=True,
                sampler=config.sampler,
                steps=config.discrete,
                maxiter=config.maxiter,
            )
            x = x.reshape(2, 2, 320, 320, 1)

            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'samples': wandb.Image(to_pil(x)),
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

    for lap in range(16):
        jobs.append(
            job(
                partial(train, runid=runid, lap=lap),
                name=f'train_{lap}',
                cpus=4,
                gpus=4,
                ram='192GB',
                time='1-00:00:00',
                partition='gpu',
            )
        )

        if len(jobs) > 1:
            jobs[-1].after(jobs[-2])

    schedule(
        *jobs,
        name=f'Training {runid}',
        backend='slurm',
        export='ALL',
        account='ariacpg',
        env=['export WANDB_SILENT=true'],
    )
