r"""Linear algebra helpers"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax import Array
from typing import *


def transpose(A: Callable[[Array], Array], x: Array) -> Callable[[Array], Array]:
    r"""Returns the transpose of a linear operation."""

    y, vjp = jax.vjp(A, x)

    def At(y):
        return next(iter(vjp(y)))

    return At


class DPLR(NamedTuple):
    r"""Diagonal plus low-rank (DPLR) matrix."""

    D: Array
    U: Array = None
    V: Array = None

    def __add__(self, C: Array) -> DPLR:  # C is scalar or diagonal
        return DPLR(self.D + C, self.U, self.V)

    def __radd__(self, C: Array) -> DPLR:
        return DPLR(C + self.D, self.U, self.V)

    def __sub__(self, C: Array) -> DPLR:
        return DPLR(self.D - C, self.U, self.V)

    def __mul__(self, C: Array) -> DPLR:
        D = self.D * C

        if self.U is None:
            U, V = None, None
        else:
            U, V = self.U, self.V * C[..., None, :]

        return DPLR(D, U, V)

    def __rmul__(self, C: Array) -> DPLR:
        D = C * self.D

        if self.U is None:
            U, V = None, None
        else:
            U, V = C[..., None] * self.U, self.V

        return DPLR(D, U, V)

    def __matmul__(self, x: Array) -> Array:
        if self.U is None:
            return self.D * x
        else:
            return self.D * x + jnp.einsum('...ij,...jk,...k', self.U, self.V, x)

    @property
    def rank(self) -> int:
        if self.U is None:
            return 0
        else:
            return self.U.shape[-1]

    @property
    def W(self) -> Array:  # capacitance
        return jnp.eye(self.rank) + jnp.einsum('...ik,...k,...kj', self.V, 1 / self.D, self.U)

    @property
    def inv(self) -> DPLR:
        D = 1 / self.D

        if self.U is None:
            U, V = None, None
        else:
            U = -D[..., None] * self.U
            V = jnp.linalg.solve(self.W, self.V) * D[..., None, :]

        return DPLR(D, U, V)

    def solve(self, x: Array) -> Array:
        D = 1 / self.D

        if self.U is None:
            return D * x
        else:
            return D * x - D * jnp.squeeze(
                self.U @ jnp.linalg.solve(self.W, self.V @ jnp.expand_dims(D * x, axis=-1)),
                axis=-1,
            )

    def diag(self) -> Array:
        if self.U is None:
            return self.D
        else:
            return self.D + jnp.einsum('...ij,...ji->...i', self.U, self.V)

    def norm(self) -> Array:
        if self.U is None:
            return jnp.sum(self.D**2, axis=-1)
        else:
            return (
                jnp.sum(self.D**2, axis=-1)
                + 2 * jnp.einsum('...i,...ij,...ji', self.D, self.U, self.V)
                + jnp.sum((self.V @ self.U) ** 2, axis=(-1, -2))
            )

    def slogdet(self) -> Tuple[Array, Array]:
        sign, logabsdet = jnp.linalg.slogdet(self.W)
        sign = sign * jnp.prod(jnp.sign(self.D), axis=-1)
        logabsdet = logabsdet + jnp.sum(jnp.log(jnp.abs(self.D)), axis=-1)

        return sign, logabsdet
