r"""Neural networks"""

import inox
import inox.nn as nn
import jax.numpy as jnp

from einops import rearrange
from inox.random import PRNG, get_rng, set_rng
from jax import Array
from typing import *


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hid_features: The number of hidden features.
        activation: The activation function constructor.
        normalize: Whether features are normalized between layers or not.
        key: A PRNG key for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hid_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = nn.SiLU,
        normalize: bool = False,
        key: Array = None,
    ):
        if key is None:
            rng = get_rng()
        else:
            rng = PRNG(key)

        layers = []

        for before, after in zip(
            (in_features, *hid_features),
            (*hid_features, out_features),
        ):
            layers.extend([
                nn.Linear(before, after, key=rng.split()),
                activation(),
                nn.LayerNorm() if normalize else None,
            ])

        layers = filter(lambda l: l is not None, layers[:-2])

        super().__init__(*layers)


class Modulation(nn.Module):
    r"""Creates an adaptive modulation module."""

    def __init__(self, channels: int, emb_features: int):
        self.mlp = nn.Sequential(
            nn.Linear(emb_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, 3 * channels),
            nn.Rearrange('... C -> ... 1 1 C'),
        )

        layer = self.mlp.layers[-2]
        layer.weight.value = layer.weight.value * 1e-1

    @inox.jit
    def __call__(self, t: Array) -> Tuple[Array, Array, Array]:
        return jnp.array_split(self.mlp(t), 3, axis=-1)


class ResBlock(nn.Module):
    r"""Creates a residual block."""

    def __init__(
        self,
        channels: int,
        emb_features: int,
        dropout: float = None,
        **kwargs,
    ):
        self.modulation = Modulation(channels, emb_features)
        self.block = nn.Sequential(
            nn.LayerNorm(),
            nn.Conv(channels, channels, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.TrainingDropout(dropout),
            nn.Conv(channels, channels, **kwargs),
        )

    def __call__(self, x: Array, t: Array) -> Array:
        a, b, c = self.modulation(t)

        y = (a + 1) * x + b
        y = self.block(y)
        y = x + c * y

        return y / jnp.sqrt(1 + c**2)


class AttBlock(nn.Module):
    r"""Creates a residual self-attention block."""

    def __init__(self, channels: int, emb_features: int, heads: int = 1):
        self.modulation = Modulation(channels, emb_features)
        self.norm = nn.LayerNorm()
        self.attn = nn.MultiheadAttention(
            heads=heads,
            in_features=channels,
            out_features=channels,
            hid_features=channels // heads,
        )

    @inox.checkpoint
    def __call__(self, x: Array, t: Array) -> Array:
        a, b, c = self.modulation(t)

        y = (a + 1) * x + b
        y = self.norm(y)
        y = rearrange(y, '... H W C -> ... (H W) C')
        y = self.attn(y)
        y = y.reshape(x.shape)
        y = x + c * y

        return y / jnp.sqrt(1 + c**2)


class UNet(nn.Module):
    r"""Creates a time (or noise) conditional U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Sequence[int] = (3, 3),
        emb_features: int = 64,
        heads: Dict[int, int] = {},
        dropout: float = None,
        key: Array = None,
    ):
        if key is None:
            key = get_rng().split()

        stride = [2 for k in kernel_size]
        kwargs = dict(
            kernel_size=kernel_size,
            padding=[(k // 2, k // 2) for k in kernel_size],
        )

        with set_rng(PRNG(key)):
            self.descent, self.ascent = [], []

            for i, blocks in enumerate(hid_blocks):
                do, up = [], []

                for _ in range(blocks):
                    do.append(ResBlock(hid_channels[i], emb_features, dropout=dropout, **kwargs))
                    up.append(ResBlock(hid_channels[i], emb_features, dropout=dropout, **kwargs))

                    if i in heads:
                        do.append(AttBlock(hid_channels[i], emb_features, heads[i]))
                        up.append(AttBlock(hid_channels[i], emb_features, heads[i]))

                if i > 0:
                    do.insert(
                        0,
                        nn.Sequential(
                            nn.Conv(
                                hid_channels[i - 1],
                                hid_channels[i],
                                stride=stride,
                                **kwargs,
                            ),
                            nn.LayerNorm(),
                        ),
                    )

                    up.append(
                        nn.Sequential(
                            nn.LayerNorm(),
                            nn.Resample(factor=stride, method='nearest'),
                        )
                    )
                else:
                    do.insert(0, nn.Conv(in_channels, hid_channels[i], **kwargs))
                    up.append(nn.Linear(hid_channels[i], out_channels))

                if i + 1 < len(hid_blocks):
                    up.insert(
                        0,
                        nn.Conv(
                            hid_channels[i] + hid_channels[i + 1],
                            hid_channels[i],
                            **kwargs,
                        ),
                    )

                self.descent.append(do)
                self.ascent.insert(0, up)

    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        r"""
        Arguments:
            x: The noisy tensor, with shape :math:`(*, H, W, C)`.
            t: The time embedding, with shape :math:`(*, T)`.
            key: A PRNG key.
        """

        if key is None:
            rng = None
        else:
            rng = PRNG(key)

        with set_rng(rng):
            memory = []

            for blocks in self.descent:
                for block in blocks:
                    if isinstance(block, (ResBlock, AttBlock)):
                        x = block(x, t)
                    else:
                        x = block(x)

                memory.append(x)

            for blocks in self.ascent:
                y = memory.pop()

                if x is not y:
                    x = jnp.concatenate((x, y), axis=-1)

                for block in blocks:
                    if isinstance(block, (ResBlock, AttBlock)):
                        x = block(x, t)
                    else:
                        x = block(x)

            return x
