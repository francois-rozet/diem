r"""CIFAR experiment helpers"""

import os

from jax import Array
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
    PATH = Path(SCRATCH) / 'diem/cifar'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def measure(A: Array, x: Array) -> Array:
    return flatten(A * unflatten(x, 32, 32))


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
        shape=flatten(y).shape,
        shard=shard,
        A=inox.Partial(measure, A),
        y=flatten(y),
        cov_y=1e-3**2,
        key=key,
        **kwargs,
    )

    x = unflatten(x, 32, 32)

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
            in_channels=3,
            out_channels=3,
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
        x = unflatten(x, width=32, height=32)
        x = super().__call__(x, t, key)
        x = flatten(x)

        return x
