r"""Optimization helpers"""

import inox
import jax
import jax.numpy as jnp
import optax

from jax import Array
from typing import *


class Adam(inox.Namespace):
    def __init__(
        self,
        steps: int,
        scheduler: str = 'constant',
        lr_init: float = 1e-3,
        lr_end: float = 1e-6,
        lr_warmup: float = 0.0,
        weight_decay: float = None,
        clip: float = None,
        **absorb,
    ):
        super().__init__(
            steps=steps,
            scheduler=scheduler,
            lr_init=lr_init,
            lr_end=lr_end,
            lr_warmup=lr_warmup,
            weight_decay=weight_decay,
            clip=clip,
        )

    def learning_rate(self, step: int) -> float:
        progress = jnp.minimum((step + 1) / (self.steps + 1), 1)
        heat = jnp.minimum((step + 1) / (self.steps * self.lr_warmup + 1), 1)

        if self.scheduler == 'constant':
            lr = self.lr_init
        elif self.scheduler == 'linear':
            lr = self.lr_init + (self.lr_end - self.lr_init) * progress
        elif self.scheduler == 'exponential':
            lr = self.lr_init * (self.lr_end / self.lr_init) ** progress

        return lr * heat

    @property
    def transform(self) -> optax.GradientTransformation:
        if self.weight_decay is None:
            optimizer = optax.adam(self.learning_rate)
        else:
            optimizer = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)

        if self.clip is None:
            return optimizer
        else:
            return optax.chain(
                optax.clip_by_global_norm(max_norm=self.clip),
                optimizer,
            )

    def init(self, *args, **kwargs) -> Any:
        return self.transform.init(*args, **kwargs)

    def update(self, *args, **kwargs) -> Any:
        return self.transform.update(*args, **kwargs)


class EMA(inox.Namespace):
    def __init__(self, decay: float = 0.999):
        self.alpha = 1.0 - decay

    def __call__(self, x: Any, y: Any) -> Any:
        return jax.tree_util.tree_map(self.average, x, y)

    def average(self, x: Array, y: Array) -> Array:
        return x + self.alpha * (y - x)
