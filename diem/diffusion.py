r"""Diffusion helpers"""

import inox
import inox.nn as nn
import jax
import jax.numpy as jnp
import numpy as np

from jax import Array
from typing import *

# isort: split
from .linalg import DPLR, transpose


class VESDE(nn.Module):
    r"""Variance exploding (VE) SDE.

    .. math:: x_t = x + \sigma_t z

    with

    .. math:: \sigma_t = \exp(\log(a) (1 - t) + \log(b) t)

    Arguments:
        a: The noise lower bound.
        b: The noise upper bound.
    """

    def __init__(self, a: Array = 1e-3, b: Array = 1e2):
        self.a = jnp.log(a)
        self.b = jnp.log(b)

    @inox.jit
    def __call__(self, x: Array, z: Array, t: Array) -> Array:
        sigma_t = self.sigma(t)
        sigma_t = sigma_t[..., None]

        return x + sigma_t * z

    @inox.jit
    def sigma(self, t: Array) -> Array:
        return jnp.exp(self.a + (self.b - self.a) * t)


class DDPM(nn.Module):
    r"""DDPM sampler for the reverse SDE.

    .. math:: x_s = x_t - \tau (x_t - f(x_t)) + \sigma_s \sqrt{\tau} \epsilon

    where :math:`\tau = 1 - \frac{\sigma_s^2}{\sigma_t^2}`.

    Arguments:
        model: A denoiser model :math:`f(x_t) \approx E[x | x_t]`.
        sde: The forward SDE.
    """

    def __init__(self, model: nn.Module, sde: VESDE = None):
        super().__init__()

        self.model = model

        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(self, xt: Array, t: Array = 1.0, steps: int = 64, key: Array = None) -> Array:
        dt = jnp.asarray(t / steps)
        time = jnp.linspace(t, dt, steps)
        keys = jax.random.split(key, steps)

        def f(xt, t_key):
            t, key = t_key
            return self.step(xt, t, t - dt, key), None

        x0, _ = jax.lax.scan(f, xt, (time, keys))

        return self.model(x0, self.sde.sigma(0.0))

    @inox.jit
    def step(self, xt: Array, t: Array, s: Array, key: Array) -> Array:
        sigma_s, sigma_t = self.sde.sigma(s), self.sde.sigma(t)
        tau = 1 - (sigma_s / sigma_t) ** 2
        eps = jax.random.normal(key, xt.shape)

        return xt - tau * (xt - self.model(xt, sigma_t)) + sigma_s * jnp.sqrt(tau) * eps


class DDIM(DDPM):
    r"""DDIM sampler for the reverse SDE.

    .. math:: x_s = x_t - (1 - \frac{\sigma_s}{\sigma_t}) (x_t - f(x_t))

    Arguments:
        model: A denoiser model :math:`f(x_t) \approx E[x | x_t]`.
        sde: The forward SDE.
    """

    @inox.jit
    def step(self, xt: Array, t: Array, s: Array, key: Array = None) -> Array:
        sigma_s, sigma_t = self.sde.sigma(s), self.sde.sigma(t)

        return xt - (1 - sigma_s / sigma_t) * (xt - self.model(xt, sigma_t))


class PredictorCorrector(DDPM):
    r"""Predictor-Corrector sampler for the reverse SDE.

    Arguments:
        model: A denoiser model :math:`f(x_t) \approx E[x | x(t)]`.
        corrections: The number of Langevin Monte Carlo (LMC) corrections.
        tau: The LMC step size.
    """

    def __init__(self, model: nn.Module, corrections: int = 1, tau: Array = 1e-1, **kwargs):
        super().__init__(model, **kwargs)

        self.corrections = corrections
        self.tau = jnp.asarray(tau)

    @inox.jit
    def step(self, xt: Array, t: Array, s: Array, key: Array) -> Array:
        xs = self.predict(xt, t, s)

        for key in jax.random.split(key, self.corrections):
            xs = self.correct(xs, s, key)

        return xs

    @inox.jit
    def predict(self, xt: Array, t: Array, s: Array) -> Array:
        sigma_s, sigma_t = self.sde.sigma(s), self.sde.sigma(t)

        return xt - (1 - sigma_s / sigma_t) * (xt - self.model(xt, sigma_t))

    @inox.jit
    def correct(self, xt: Array, t: Array, key: Array) -> Array:
        sigma_t = self.sde.sigma(t)
        eps = jax.random.normal(key, xt.shape)

        return xt - self.tau * (xt - self.model(xt, sigma_t)) + sigma_t * jnp.sqrt(2 * self.tau) * eps


class PosEmbedding(nn.Module):
    r"""Creates a positional embedding module.

    References:
        | Attention Is All You Need (Vaswani et al., 2017)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        freqs = np.linspace(0, 1, features // 2)
        freqs = (1 / 1e4) ** freqs

        self.freqs = jnp.asarray(freqs)

    @inox.jit
    def __call__(self, x: Array) -> Array:
        x = x[..., None]

        return jnp.concatenate(
            (
                jnp.sin(self.freqs * x),
                jnp.cos(self.freqs * x),
            ),
            axis=-1,
        )


class Denoiser(nn.Module):
    r"""Denoiser model with EDM-style preconditioning.

    .. math:: f(x_t) \approx E[x | x_t]

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        network: A noise conditional network.
    """

    def __init__(self, network: nn.Module, emb_features: int = 64):
        self.net = network
        self.emb = PosEmbedding(emb_features)

    @inox.jit
    def __call__(self, xt: Array, sigma_t: Array, key: Array = None) -> Array:
        r"""
        Arguments:
            xt: The noisy tensor, with shape :math:`(*, D)`.
            sigma_t: The noise std, with shape :math:`(*)`.
            key: A PRNG key.
        """

        c_skip = 1 / (sigma_t**2 + 1)
        c_out = sigma_t / jnp.sqrt(sigma_t**2 + 1)
        c_in = 1 / jnp.sqrt(sigma_t**2 + 1)
        c_noise = jnp.log(sigma_t)

        c_skip, c_out, c_in = c_skip[..., None], c_out[..., None], c_in[..., None]

        return c_skip * xt + c_out * self.net(c_in * xt, self.emb(c_noise), key)


class DenoiserLoss(nn.Module):
    r"""Loss for a denoiser model.

    .. math:: \lambda_t || A f(x_t) - y ||^2

    Arguments:
        sde: The forward SDE.
    """

    def __init__(self, sde: VESDE = None):
        if sde is None:
            self.sde = VESDE()
        else:
            self.sde = sde

    @inox.jit
    def __call__(
        self,
        model: nn.Module,
        x: Array,
        z: Array,
        t: Array,
        A: Callable[[Array], Array] = None,  # /!\ linear
        y: Array = None,
        key: Array = None,
    ) -> Array:
        sigma_t = self.sde.sigma(t)
        lmbda_t = 1 / sigma_t**2 + 1

        xt = self.sde(x, z, t)
        ft = model(xt, sigma_t, key)

        if A is None:
            A = lambda x: x

        if y is None:
            y = A(x)

        error = A(ft) - y

        return jnp.mean(lmbda_t * jnp.mean(error**2, axis=-1))


class GaussianDenoiser(nn.Module):
    r"""Denoiser model for a Gaussian random variable.

    .. math:: p(x) = N(x | \mu_x, \Sigma_x)

    Arguments:
        mu_x: The mean :math:`\mu_x`.
        cov_x: The covariance :math:`\Sigma_x`.
    """

    def __init__(
        self,
        mu_x: Array = 0.0,
        cov_x: Union[Array, DPLR] = 1.0,
    ):
        if not isinstance(cov_x, DPLR):
            cov_x = DPLR(cov_x)

        self.mu_x = jnp.asarray(mu_x)
        self.cov_x = jax.tree_util.tree_map(jnp.asarray, cov_x)

    @inox.jit
    def __call__(self, xt: Array, sigma_t: Array, key: Array = None) -> Array:
        cov_t = sigma_t[..., None] ** 2

        return xt - cov_t * (self.cov_x + cov_t).solve(xt - self.mu_x)


class PosteriorDenoiser(nn.Module):
    r"""Posterior denoiser model for a Gaussian observation.

    .. math:: p(y | x) = N(y | Ax, \Sigma_y)

    Arguments:
        model: A denoiser model :math:`f(x_t) \approx E[x | x_t]`.
        A: The forward model :math:`A`.
        y: An observation.
        cov_y: The observation covariance :math:`\Sigma_y`.
        cov_x: The hidden covariance :math:`\Sigma_x`.
    """

    def __init__(
        self,
        model: nn.Module,
        A: Callable[[Array], Array],
        y: Array,
        cov_y: Union[Array, DPLR],
        cov_x: Union[Array, DPLR] = None,
        rtol: float = 1e-3,
        maxiter: int = 1,
        method: str = 'cg',
        verbose: bool = False,
    ):
        super().__init__()

        self.model = model
        self.A = A
        self.y = jnp.asarray(y)

        if not isinstance(cov_y, DPLR):
            cov_y = DPLR(cov_y)

        if not isinstance(cov_x, DPLR) and cov_x is not None:
            cov_x = DPLR(cov_x)

        self.cov_y = jax.tree_util.tree_map(jnp.asarray, cov_y)
        self.cov_x = jax.tree_util.tree_map(jnp.asarray, cov_x)

        self.rtol = rtol
        self.maxiter = maxiter

        if method == 'cg':
            self.solve = jax.scipy.sparse.linalg.cg
        elif method == 'bicgstab':
            self.solve = jax.scipy.sparse.linalg.bicgstab

        self.verbose = verbose

    @inox.jit
    def __call__(self, xt: Array, sigma_t: Array, key: Array = None) -> Array:
        cov_t = sigma_t[..., None] ** 2

        x, vjp = jax.vjp(lambda xt: self.model(xt, sigma_t, key), xt)
        y, A = jax.linearize(self.A, x)
        At = transpose(A, x)

        if self.cov_x is None:
            cov_y_xt = lambda v: self.cov_y @ v + cov_t * A(*vjp(At(v)))
        else:
            cov_x_xt = cov_t + (-(cov_t**2)) * (self.cov_x + cov_t).inv
            cov_y_xt = lambda v: self.cov_y @ v + A(cov_x_xt @ At(v))

        b = self.y - y
        v, _ = self.solve(
            A=cov_y_xt,
            b=b,
            tol=self.rtol,
            maxiter=self.maxiter,
        )

        if self.verbose:
            jax.debug.print('{},{}', sigma_t, jnp.linalg.norm(cov_y_xt(v) - b))

        (score,) = vjp(At(v))

        return x + cov_t * score
