r"""Common helpers"""

import inox
import inox.nn as nn
import jax
import jax.experimental.sparse as jes
import jax.numpy as jnp
import pickle

from inox.random import PRNG, get_rng
from jax import Array
from pathlib import Path
from tqdm import tqdm
from typing import *

# isort: split
from .diffusion import *
from .linalg import *


def dump_module(module: nn.Module, file: Path):
    with open(file, 'wb') as f:
        pickle.dump(module, f)


def load_module(file: Path) -> nn.Module:
    with open(file, 'rb') as f:
        return pickle.load(f)


def distribute(tree: Any) -> Any:
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    spec = jax.sharding.PartitionSpec('i')
    dist = jax.sharding.NamedSharding(mesh, spec)

    return jax.device_put(tree, dist)


@inox.jit
def ppca(x: Array, key: Array, rank: int = 1) -> Tuple[Array, DPLR]:
    r"""Fits :math:`(\mu_x, \Sigma_x)` by probabilistic principal component analysis (PPCA).

    References:
        https://www.miketipping.com/papers/met-mppca.pdf
    """

    samples, features = x.shape

    mu_x = jnp.mean(x, axis=0)
    x = x - mu_x

    if samples < features:
        C = x @ x.T / samples
    else:
        C = x.T @ x / samples

    if rank < len(C) // 5:
        Q = jax.random.normal(key, (len(C), rank))
        L, Q, _ = jes.linalg.lobpcg_standard(C, Q)
    else:
        L, Q = jnp.linalg.eigh(C)
        L, Q = L[-rank:], Q[:, -rank:]

    if samples < features:
        Q = x.T @ Q
        Q = Q / jnp.linalg.norm(Q, axis=0)

    if rank < features:
        D = (jnp.trace(C) - jnp.sum(L)) / (features - rank)
    else:
        D = jnp.asarray(1e-6)

    U = Q * jnp.sqrt(jnp.maximum(L - D, 0.0))

    cov_x = DPLR(D * jnp.ones(features), U, U.T)

    return mu_x, cov_x


def fit_moments(
    features: int,
    rank: int,
    A: Callable[[Array], Array],
    y: Array,
    cov_y: Tuple[Array, DPLR],
    iterations: int = 16,
    key: Array = None,
    **kwargs,
) -> Tuple[Array, DPLR]:
    r"""Fits :math:`(\mu_x, \Sigma_x)` by expectation maximization."""

    if key is None:
        rng = get_rng()
    else:
        rng = PRNG(key)

    mu_x = jnp.zeros(features)
    cov_x = DPLR(
        jnp.ones(features),
        jnp.zeros((features, rank)),
        jnp.zeros((rank, features)),
    )

    for _ in tqdm(range(iterations), ncols=88):
        # Expectation
        x = sample_any(
            model=GaussianDenoiser(mu_x, cov_x),
            shape=(len(y), features),
            A=A,
            y=y,
            cov_y=cov_y,
            key=rng.split(),
            **kwargs,
        )

        # Maximization
        mu_x, cov_x = ppca(x, rank=rank, key=rng.split())

    return mu_x, cov_x


def sample_any(
    model: nn.Module,
    shape: Sequence[int],
    shard: bool = False,
    A: Callable[[Array], Array] = None,
    y: Array = None,
    cov_y: Union[Array, DPLR] = None,
    key: Array = None,
    sampler: str = 'ddpm',
    steps: int = 64,
    rtol: float = 1e-3,
    maxiter: int = 1,
    method: str = 'cg',
    verbose: bool = False,
    **kwargs,
) -> Array:
    r"""Samples from :math:`q(x)` or :math:`q(x | A, y)`."""

    mu_x = getattr(model, 'mu_x', None)
    cov_x = getattr(model, 'cov_x', None)

    if A is None or y is None:
        pass
    else:
        model = PosteriorDenoiser(
            model=model,
            A=A,
            y=y,
            cov_y=cov_y,
            cov_x=cov_x,
            rtol=rtol,
            maxiter=maxiter,
            method=method,
            verbose=verbose,
        )

    if sampler == 'ddpm':
        sampler = DDPM(model, **kwargs)
    elif sampler == 'ddim':
        sampler = DDIM(model, **kwargs)
    elif sampler == 'pc':
        sampler = PredictorCorrector(model, **kwargs)

    z = jax.random.normal(key, shape)

    if shard:
        z = distribute(z)

    if mu_x is None:
        x1 = sampler.sde(0.0, z, 1.0)
    else:
        x1 = sampler.sde(mu_x, z, 1.0)

    x0 = sampler(x1, steps=steps, key=key)

    return x0
