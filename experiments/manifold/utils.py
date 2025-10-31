r"""Manifold experiment helpers"""

import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import os
import ot
import pandas as pd
import seaborn as sb

from functools import partial
from jax import Array
from typing import *

# isort: split
from diem.common import *
from diem.diffusion import *
from diem.image import *
from diem.nn import *
from diem.optim import *

if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'diem/manifold'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def measure(A: Array, x: Array) -> Array:
    return jnp.einsum('...ij,...j', A, x)


def show_pair(y: Array, cmap: str = 'Blues', **kwargs) -> plt.Figure:
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in range(16, cmap.N)]
    colors = [(1.0, 1.0, 1.0), *colors]
    cmap = plt.cm.colors.ListedColormap(colors)

    return sb.histplot(
        data=pd.DataFrame({'$y_0$': y[:, 0], '$y_1$': y[:, 1]}),
        x='$y_0$',
        y='$y_1$',
        bins=64,
        binrange=(-3, 3),
        thresh=None,
        cmap=cmap,
        **kwargs,
    )


def show_corner(x: Array, cmap: str = 'Blues', **kwargs) -> plt.Figure:
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in range(16, cmap.N)]
    colors = [(1.0, 1.0, 1.0), *colors]
    cmap = plt.cm.colors.ListedColormap(colors)

    return sb.pairplot(
        data=pd.DataFrame({f'$x_{i}$': xi for i, xi in enumerate(np.asarray(x).T)}),
        corner=True,
        kind='hist',
        plot_kws={'bins': 64, 'binrange': (-3, 3), 'thresh': None, 'cmap': cmap},
        diag_kws={'bins': 64, 'binrange': (-3, 3), 'element': 'step', 'color': cmap(cmap.N // 2)},
        **kwargs,
    )


def sinkhorn_divergence(
    u1: Array,
    u2: Array,
    v: Array,
    lmbda: float = 1e-3,
    maxiter: int = 1024,
    epsilon: float = 1e-3,
) -> Array:
    r"""Computes the Sinkhorn divergence between two samples.

    References:
        | Faster Wasserstein Distance Estimation with the Sinkhorn Divergence (Chizat et al., 2020)
        | https://arxiv.org/abs/2006.08172
    """

    def transport(u, v):
        return ot.sinkhorn2(
            a=jnp.asarray(()),
            b=jnp.asarray(()),
            M=ot.dist(u, v),
            reg=lmbda,
            numItermax=maxiter,
            stopThr=epsilon,
            method='sinkhorn_log',
        )

    return jnp.maximum(transport(u1, v) - transport(u1, u2), 1e-6)


def smooth_manifold(
    key: Array,
    shape: Sequence[int] = (),
    m: int = 1,
    n: int = 3,
    alpha: float = 3.0,
    epsilon: float = 1e-3,
) -> Array:
    r"""Samples points from a smooth random manifold.

    References:
        https://github.com/fzenke/randman

    Arguments:
        m: The manifold dimension.
        n: The space dimension.
        alpha: The smoothness coefficient.
    """

    key_params, key_z = jax.random.split(key, 2)

    cutoff = math.ceil(epsilon ** (-1 / alpha))
    k = jnp.arange(cutoff) + 1

    a, b, c = jax.random.uniform(key_params, (3, n, m, cutoff))

    z = jax.random.uniform(key_z, (*shape, 1, m, 1))
    x = a / k**alpha * jnp.sin(2 * jnp.pi * (k * b * z + c))
    x = jnp.sum(x, axis=-1)
    x = jnp.prod(x, axis=-1)

    return x


def make_model(
    key: Array,
    features: int,
    hid_features: Sequence[int] = (256, 256, 256),
    emb_features: int = 64,
    normalize: bool = True,
    **absorb,
) -> Denoiser:
    return Denoiser(
        network=TimeMLP(
            features=features,
            hid_features=hid_features,
            emb_features=emb_features,
            normalize=normalize,
            key=key,
        ),
        emb_features=emb_features,
    )


class TimeMLP(MLP):
    def __init__(self, features: int, emb_features: int = 64, **kwargs):
        super().__init__(features + emb_features, features, **kwargs)

    @staticmethod
    @partial(jnp.vectorize, signature='(m),(n)->(p)')
    def cat(x: Array, y: Array) -> Array:
        return jnp.concatenate((x, y))

    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        return super().__call__(self.cat(x, t))
