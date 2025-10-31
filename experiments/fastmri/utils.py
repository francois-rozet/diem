r"""FastMRI experiment helpers"""

import math
import os

from jax import Array
from jax.experimental.shard_map import shard_map
from pathlib import Path
from typing import *

# isort: split
from diem.common import *
from diem.data import *
from diem.diffusion import *
from diem.image import *
from diem.nn import *
from diem.optim import *

if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'diem/fastmri'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def real2complex(x: Array) -> Array:
    return jax.lax.complex(*jnp.array_split(x, 2, axis=-1))


def complex2real(x: Array) -> Array:
    return jnp.concatenate((x.real, x.imag), axis=-1)


def fft2c(x: Array, norm: str = 'ortho') -> Array:
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.ifftshift(x, axes=(-3, -2)),
            axes=(-3, -2),
            norm=norm,
        ),
        axes=(-3, -2),
    )


def ifft2c(k: Array, norm: str = 'ortho') -> Array:
    return jnp.fft.fftshift(
        jnp.fft.ifft2(
            jnp.fft.ifftshift(
                k,
                axes=(-3, -2),
            ),
            axes=(-3, -2),
            norm=norm,
        ),
        axes=(-3, -2),
    )


def make_mask(r: int = 4, key: Array = None) -> Array:
    r"""Creates an horizontal frequency subsampling mask.

    References:
        | Robust Compressed Sensing MRI with Deep Generative Priors
        | https://arxiv.org/abs/2108.01368
    """

    if key is None:
        A = np.random.uniform(size=(1, 320, 1))
        A = jnp.asarray(A)
    else:
        A = jax.random.uniform(key, shape=(1, 320, 1))

    A = A < 200 / (320 * r - 120)
    A = A.at[:, 160 - math.ceil(60 / r) : 160 + math.ceil(60 / r)].set(True)

    return A


def measure(A: Array, x: Array, shard: bool = False) -> Array:
    def f(A: Array, x: Array) -> Array:
        x = unflatten(x, 320, 320)
        y = fft2c(x)
        y = A * y
        y = complex2real(y)
        y = flatten(y)

        return y

    if shard:
        mesh = jax.sharding.Mesh(jax.devices(), 'i')
        spec = jax.sharding.PartitionSpec('i')

        return shard_map(
            f=f,
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
        )(A, x)
    else:
        return f(A, x)


def sample(
    model: nn.Module,
    y: Array,
    A: Array,
    key: Array,
    shard: bool = False,
    **kwargs,
) -> Array:
    if shard:
        y, A = distribute((y, A))

    x = sample_any(
        model=model,
        shape=(len(y), 320 * 320 * 1),
        shard=shard,
        A=inox.Partial(measure, A, shard=shard),
        y=flatten(y),
        cov_y=1e-2**2,
        key=key,
        **kwargs,
    )

    x = unflatten(x, 320, 320)

    return x


def make_model(
    key: Array,
    hid_channels: Sequence[int] = (64, 128, 256),
    hid_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    heads: Dict[int, int] = {2: 1},
    dropout: float = None,
    **absorb,
) -> Denoiser:
    return Denoiser(
        network=FlatUNet(
            in_channels=16,
            out_channels=16,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            emb_features=emb_features,
            heads=heads,
            dropout=dropout,
            key=key,
        ),
        emb_features=emb_features,
    )


class FlatUNet(UNet):
    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x, width=320, height=320)
        x = rearrange(x, '... (H h) (W w) C -> ... H W (h w C)', h=4, w=4)
        x = super().__call__(x, t, key)
        x = rearrange(x, '... H W (h w C) -> ... (H h) (W w) C', h=4, w=4)
        x = flatten(x)

        return x
