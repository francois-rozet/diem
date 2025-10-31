#!/usr/bin/env python

import jax
import jax.numpy as jnp
import wandb

from dawgz import job, schedule
from typing import *

# isort: split
from utils import *

CONFIG = {
    'm': 1,
    'n': 3,
    'alpha': 3.0,
    'components': 256,
    'thickness': 1e-2,
    'p': 1,
    'noise': 1e-2,
    'samples': 2**20,
}


def evaluate(**config):
    run = wandb.init(project='diem-manifold', dir=PATH, config=config)

    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    config = run.config

    # Precision
    jax.config.update('jax_enable_x64', True)

    # RNG
    rng = PRNG(config.seed)

    # Manifold
    mu_i = smooth_manifold(
        key=rng.split(),
        shape=(config.components,),
        m=config.m,
        n=config.n,
        alpha=config.alpha,
    )

    mu_i = (mu_i - mu_i.min(axis=0)) / (mu_i.max(axis=0) - mu_i.min(axis=0))
    mu_i = 4.0 * mu_i - 2.0

    sigma_i = config.thickness * jnp.ones_like(mu_i)

    # Observation
    W = rng.normal((config.p, config.n))
    W = W / jnp.linalg.norm(W, axis=-1, keepdims=True)

    def A(x):
        return jnp.einsum('...ij,...j', W, x)

    cov_y = config.noise**2 * jnp.eye(config.p)

    x = mu_i[0]
    y = rng.multivariate_normal(A(x), cov_y)

    # Prior
    i = rng.randint((config.samples,), minval=0, maxval=config.components)
    p_x = mu_i[i] + sigma_i[i] * rng.normal((config.samples, config.n))

    cov_x = jnp.cov(p_x.mT)

    prior = show_corner(p_x)
    prior.save(runpath / 'prior.png')

    run.log({
        'prior': wandb.Image(prior),
    })

    # Posterior
    log_p_y_x = jax.scipy.stats.multivariate_normal.logpdf(y, A(p_x), cov_y)

    p_x_y = rng.choice(p_x, (config.samples,), p=jnp.exp(log_p_y_x))

    posterior = show_corner(p_x_y)
    posterior.save(runpath / 'posterior.png')

    # Noisy posterior(s)
    @jax.jit
    def log_prior(xt, sigma_t):
        log_pi = jax.scipy.stats.norm.logpdf(xt[..., None, :], mu_i, jnp.sqrt(sigma_i**2 + sigma_t**2))
        log_pi = jnp.sum(log_pi, axis=-1)

        return jax.scipy.special.logsumexp(log_pi, axis=-1)

    @jax.jit
    def log_likelihood(xt, sigma_t):
        J = jnp.vectorize(jax.jacobian(log_prior), signature='(n),()->(n)')(xt, sigma_t)
        E_x_xt = xt + sigma_t**2 * J
        E_y_xt = A(E_x_xt)

        if config.heuristic == 'zero':
            V_x_xt = 0.0 * jnp.eye(config.n)
        elif config.heuristic == 'cov_t':
            V_x_xt = sigma_t**2 * jnp.eye(config.n)
        else:
            if config.heuristic == 'cov_x':
                H = -jnp.linalg.inv(cov_x + sigma_t**2 * jnp.eye(config.n))
            elif config.heuristic == 'hessian':
                H = jnp.vectorize(jax.hessian(log_prior), signature='(n),()->(n,n)')(xt, sigma_t)

            V_x_xt = sigma_t**2 * jnp.eye(config.n) + sigma_t**4 * H

        V_y_xt = cov_y + A(A(V_x_xt).mT)

        return jax.scipy.stats.multivariate_normal.logpdf(y, E_y_xt, V_y_xt)

    for sigma_t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        p_xt = p_x + sigma_t * rng.normal(p_x.shape)

        log_q_y_xt = [log_likelihood(xt, sigma_t) for xt in jnp.array_split(p_xt, 16, axis=0)]
        log_q_y_xt = jnp.concatenate(log_q_y_xt)

        q_xt_y = rng.choice(p_xt, (config.samples,), p=jnp.exp(log_q_y_xt))

        divergence = sinkhorn_divergence(
            u1=p_x_y[:16384] + sigma_t * rng.normal(p_x_y[:16384].shape),
            u2=p_x_y[-16384:] + sigma_t * rng.normal(p_x_y[-16384:].shape),
            v=q_xt_y[:16384],
        )

        run.log({
            'sigma': sigma_t,
            'posterior': wandb.Image(show_corner(q_xt_y)),
            'divergence': divergence,
        })


if __name__ == '__main__':
    jobs = []

    for seed in range(64):
        for heuristic in ('zero', 'cov_t', 'cov_x', 'hessian'):
            jobs.append(
                job(
                    partial(
                        evaluate,
                        heuristic=heuristic,
                        seed=seed,
                        **CONFIG,
                    ),
                    name=f'eval_{seed}_{heuristic}',
                    cpus=4,
                    gpus=1,
                    ram='16GB',
                    time='01:00:00',
                    partition='gpu',
                )
            )

    schedule(
        *jobs,
        name='Comparison',
        backend='slurm',
        export='ALL',
        account='ariacpg',
        env=['export WANDB_SILENT=true'],
    )
