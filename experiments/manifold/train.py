#!/usr/bin/env python

import inox
import inox.nn as nn
import jax
import numpy as np
import optax
import wandb

from dawgz import job, schedule
from tqdm import tqdm
from typing import *

# isort: split
from utils import *

CONFIG = {
    # Data
    'seed': 0,
    'samples': 65536,
    'features': 5,
    'observe': 2,
    'noise': 1e-2,
    # Architecture
    'hid_features': (256, 256, 256),
    'emb_features': 64,
    'normalize': True,
    # Sampling
    'sampler': 'pc',
    'heuristic': 'cov_x',
    'sde': {'a': 1e-3, 'b': 1e1},
    'discrete': 4096,
    'maxiter': None,
    # Training
    'laps': 64,
    'epochs': 65536,
    'batch_size': 1024,
    'scheduler': 'linear',
    'lr_init': 1e-3,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
}


@job(cpus=4, gpus=1, ram='16GB', time='06:00:00', partition='gpu')
def train():
    run = wandb.init(
        project='diem-manifold-linear',
        dir=PATH,
        config=CONFIG,
    )

    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # RNG
    seed = hash(runpath) % 2**16
    rng = inox.random.PRNG(seed)

    # SDE
    sde = VESDE(**CONFIG.get('sde'))

    # Data
    keys = jax.random.split(jax.random.key(config.seed))

    ## Latent
    x = smooth_manifold(keys[0], shape=(config.samples,), m=1, n=config.features)
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    x = 4.0 * x - 2.0

    ## Observations
    A = jax.random.normal(keys[1], (config.samples, config.observe, config.features))
    A = A / jnp.linalg.norm(A, axis=-1, keepdims=True)

    cov_y = config.noise**2 * jnp.ones(config.observe)

    y = measure(A, x) + jnp.sqrt(cov_y) * rng.normal((config.samples, config.observe))

    ## Moments
    mu_x, cov_x = fit_moments(
        features=config.features,
        rank=config.features,
        A=inox.Partial(measure, A),
        y=y,
        cov_y=cov_y,
        sampler='ddim',
        sde=sde,
        steps=256,
        maxiter=None,
        key=rng.split(),
    )

    ## pi_0
    def generate(model: nn.Module, **kwargs) -> Array:
        def fun(A: Array, y: Array, key: Array) -> Array:
            return sample_any(
                model=model,
                shape=(len(y), config.features),
                A=inox.Partial(measure, A),
                y=y,
                cov_y=cov_y,
                sampler=config.sampler,
                sde=sde,
                steps=config.discrete,
                maxiter=config.maxiter,
                key=key,
                **kwargs,
            )

        x = jax.vmap(fun)(
            rearrange(A, '(M N) ... -> M N ...', M=256),
            rearrange(y, '(M N) ... -> M N ...', M=256),
            rng.split(256),
        )

        return rearrange(x, 'M N ... -> (M N) ...')

    pi = generate(GaussianDenoiser(mu_x, cov_x))

    # Model
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
    optimizer = Adam(steps=config.epochs, **config)
    opt_state = optimizer.init(params)

    # Training
    @jax.jit
    def ell(params, others, x, key):
        keys = jax.random.split(key, 3)

        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])

        return objective(static(params, others), x, z, t, key=keys[2])

    @jax.jit
    def sgd_step(params, others, opt_state, x, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, opt_state

    for lap in tqdm(range(config.laps), ncols=88):
        ## SGD
        losses = []

        for epoch in range(config.epochs):
            i = rng.randint(shape=(config.batch_size,), minval=0, maxval=len(pi))
            loss, params, opt_state = sgd_step(params, others, opt_state, pi[i], rng.split())

            losses.append(loss)

        losses = np.stack(losses)

        ## Eval
        model = static(params, others)
        model.train(False)

        pi = generate(model)
        mask = jnp.all(jnp.logical_and(-3 < pi, pi < 3), axis=-1)
        pi = pi[mask]

        np.save(runpath / f'checkpoint_{lap}.npy', pi)

        divergence = sinkhorn_divergence(
            x[:16384],
            x[-16384:],
            pi[:16384],
        )

        fig = show_corner(pi)._figure

        run.log({
            'loss': np.mean(losses),
            'loss_std': np.std(losses),
            'divergence': divergence,
            'corner': wandb.Image(fig),
        })

        ## Restart
        opt_state = optimizer.init(params)


if __name__ == '__main__':
    schedule(
        train,
        name='Training',
        backend='slurm',
        export='ALL',
        account='ariacpg',
        env=['export WANDB_SILENT=true'],
    )
