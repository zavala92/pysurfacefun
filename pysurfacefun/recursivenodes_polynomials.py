"""RecursiveNodes Proriol-Koornwinder-Dubiner polynomial utilities.

This module is a small, local adaptation of the polynomial routines from
Tobin Isaac's ``recursivenodes`` package.  The PKD polynomials are orthonormal
on the biunit simplex.  The triangular surface code maps unit triangle
coordinates to the biunit triangle before evaluating these routines.
"""

from __future__ import annotations

from math import comb, lgamma

import numpy as np


Array = np.ndarray


def npolys(d: int, k: int) -> int:
    """Number of polynomials of total degree at most ``k`` in ``d`` dimensions."""
    return comb(k + d, d)


def multiindex_equal(d: int, k: int):
    """Yield ``d``-tuples of nonnegative integers whose sum is ``k``."""
    if d <= 0 or k < 0:
        return
    for i in range(k):
        for alpha in multiindex_equal(d - 1, k - i):
            yield (i,) + alpha
    yield (k,) + (0,) * (d - 1)


def multiindex_up_to(d: int, k: int):
    """Yield ``d``-tuples of nonnegative integers whose sum is at most ``k``."""
    for alpha in multiindex_equal(d + 1, k):
        yield alpha[:d]


def _jacobi_recurrence(n, alpha: float = 0.0, beta: float = 0.0) -> tuple[Array, Array, Array]:
    """Three-term recurrence coefficients for Jacobi polynomials."""
    n = np.asarray(n, dtype=int)
    a = np.zeros(n.shape)
    b = np.zeros(n.shape)
    c = np.zeros(n.shape)
    a[n == 1] = (alpha + beta + 2) / 2
    b[n == 1] = (alpha - beta) / 2
    N = n[n > 1]
    a[n > 1] = (2 * N + alpha + beta) * (2 * N + alpha + beta - 1) / (
        2 * N * (N + alpha + beta)
    )
    b[n > 1] = ((alpha * alpha - beta * beta) * (2 * N + alpha + beta - 1)) / (
        2 * N * (N + alpha + beta) * (2 * N + alpha + beta - 2)
    )
    c[n > 1] = (2 * (N + alpha - 1) * (N + beta - 1) * (2 * N + alpha + beta)) / (
        2 * N * (N + alpha + beta) * (2 * N + alpha + beta - 2)
    )
    return a, b, c


def _eval_jacobi(n: int, alpha: float, beta: float, x: Array, out: Array | None = None) -> Array:
    """Pure NumPy replacement for ``scipy.special.eval_jacobi`` for integer ``n``."""
    x = np.asarray(x)
    if out is None:
        out = np.ones(x.shape)
    else:
        out[:] = 1.0
    n = int(n)
    if n == 0:
        return out

    a, b, c = _jacobi_recurrence(list(range(1, n + 1)), alpha, beta)
    pm1 = np.zeros(out.shape)
    pm2 = np.empty(out.shape)
    for i in range(n):
        pm2[...] = pm1[...]
        pm1[...] = out[...]
        out[...] = (a[i] * x + b[i]) * pm1 - c[i] * pm2
    return out


try:
    from scipy.special import eval_jacobi as _scipy_eval_jacobi
except Exception:  # pragma: no cover - SciPy is optional here.
    _scipy_eval_jacobi = None


def rn_jacobi(n: int, x: Array, a: float = 0.0, b: float = 0.0, out: Array | None = None) -> Array:
    """Evaluate a Jacobi polynomial using the RecursiveNodes convention."""
    if _scipy_eval_jacobi is not None:
        return _scipy_eval_jacobi(n, a, b, x, out)
    return _eval_jacobi(n, a, b, x, out)


def rn_jacobider(
    n: int,
    x: Array,
    a: float = 0.0,
    b: float = 0.0,
    k: int = 1,
    out: Array | None = None,
) -> Array:
    """Evaluate the ``k``-th derivative of a Jacobi polynomial."""
    if k > n:
        if out is not None:
            out[:] = 0.0
            return out
        return np.zeros(np.shape(x))
    return rn_jacobi(n - k, x, a + k, b + k) * np.exp(
        lgamma(a + b + n + 1 + k) - lgamma(a + b + n + 1)
    ) / 2**k


def rn_jacobinorm2(n: int, a: float = 0.0, b: float = 0.0) -> float:
    """Weighted square ``L2`` norm of a Jacobi polynomial on ``[-1, 1]``."""
    if n == 0:
        return 2.0 ** (a + b + 1) * np.exp(lgamma(a + 1) + lgamma(b + 1) - lgamma(a + b + 2))
    return (
        2.0 ** (a + b + 1)
        / (2 * n + a + b + 1)
        * np.exp(lgamma(n + a + 1) + lgamma(n + b + 1) - lgamma(n + a + b + 1) - lgamma(n + 1))
    )


def proriolkoornwinderdubiner(d: int, i: tuple[int, ...], x: Array, out: Array | None = None) -> Array:
    """Evaluate one orthonormal PKD polynomial on the biunit simplex."""
    x = np.asarray(x)
    if d == 1:
        return rn_jacobi(i[0], x[:, 0], out=out) / rn_jacobinorm2(i[0]) ** 0.5

    isum = sum(i[: d - 1])
    factor = 1.0 - x[:, d - 1]
    nonzero = np.abs(factor) > 1.0e-10
    if out is None:
        px = np.empty(x[:, : d - 1].shape)
    else:
        px = out
    px[nonzero, :] = ((x[nonzero, : d - 1] + 1.0) * 2.0 / factor[nonzero, np.newaxis] - 1.0)
    px[~nonzero, :] = -1.0

    pi = proriolkoornwinderdubiner(d - 1, i[: d - 1], px)
    pi *= rn_jacobi(i[-1], x[:, d - 1], 2 * isum + d - 1, 0.0)
    pi /= rn_jacobinorm2(i[-1], 2 * isum + d - 1, 0.0) ** 0.5
    pi *= factor**isum
    pi *= 2.0 ** ((d - 1) / 2)
    return pi


def proriolkoornwinderdubinergrad(
    d: int,
    i: tuple[int, ...],
    x: Array,
    out: Array | None = None,
    both: bool = False,
):
    """Evaluate the gradient of one PKD polynomial on the biunit simplex."""
    x = np.asarray(x)
    if d == 1:
        jnorm = rn_jacobinorm2(i[0]) ** 0.5
        if both:
            pkd = rn_jacobi(i[0], x[:, 0]) / jnorm
        grad = rn_jacobider(i[0], x[:, 0]).reshape(x.shape)
        grad /= jnorm
        if both:
            return pkd, grad
        return grad

    isum = sum(i[: d - 1])
    factor = 1.0 - x[:, d - 1]
    nonzero = np.abs(factor) > 1.0e-10
    px = np.empty(x[:, : d - 1].shape)
    px[nonzero, :] = ((x[nonzero, : d - 1] + 1.0) * 2.0 / factor[nonzero, np.newaxis] - 1.0)
    px[~nonzero, :] = -1.0

    pxgradfisum = np.zeros(px.shape + (d,))
    for j in range(d - 1):
        pxgradfisum[nonzero, j, j] = 2.0 * factor[nonzero] ** (isum - 1)
        pxgradfisum[nonzero, j, d - 1] = (
            (x[nonzero, j] + 1.0) * 2.0 * factor[nonzero] ** (isum - 2)
        )
        if isum > 0:
            pxgradfisum[~nonzero, j, j] = 2.0 * factor[~nonzero] ** (isum - 1)
        if isum > 1:
            pxgradfisum[~nonzero, j, d - 1] = (
                (x[~nonzero, j] + 1.0) * 2.0 * factor[~nonzero] ** (isum - 2)
            )

    pi, pigrad = proriolkoornwinderdubinergrad(d - 1, i[: d - 1], px, both=True)
    pz = rn_jacobi(i[-1], x[:, d - 1], 2 * isum + d - 1, 0.0)
    jnorm = rn_jacobinorm2(i[-1], 2 * isum + d - 1, 0.0) ** 0.5
    pz /= jnorm
    pzgrad = rn_jacobider(i[-1], x[:, d - 1], 2 * isum + d - 1, 0.0)
    pzgrad /= jnorm
    fisum = factor**isum

    if out is not None:
        pkdgrad = out
        pkdgrad[:] = 0.0
    else:
        pkdgrad = np.zeros(x.shape)
    if both:
        pkd = pi * pz * fisum * 2.0 ** ((d - 1) / 2)

    pkdgrad[:, :] = np.einsum("ij,ijk->ik", pigrad, pxgradfisum).reshape(x.shape) * pz[:, np.newaxis]
    pkdgrad[:, d - 1] += pi * pzgrad * fisum
    if isum != 0:
        pkdgrad[:, d - 1] -= isum * pi * pz * factor ** (isum - 1)
    pkdgrad *= 2.0 ** ((d - 1) / 2)

    if both:
        return pkd, pkdgrad
    return pkdgrad


def proriolkoornwinderdubinervandermonde(d: int, n: int, x: Array, out: Array | None = None, C: Array | None = None) -> Array:
    """Evaluate the PKD Vandermonde matrix on the biunit simplex."""
    x = np.asarray(x)
    N = npolys(d, n)
    if out is not None:
        V = out
    elif C is None:
        V = np.empty(x.shape[:-1] + (N,))
    else:
        V = np.empty(x.shape[:-1] + (C.shape[1],))

    if C is not None:
        V[:] = 0.0
    for k, i in enumerate(multiindex_up_to(d, n)):
        pkd = proriolkoornwinderdubiner(d, i, x)
        if C is None:
            V[:, k] = pkd
        else:
            V[:, :] += np.einsum("i,j->ij", pkd, C[k, :])
    return V


def proriolkoornwinderdubinervandermondegrad(
    d: int,
    n: int,
    x: Array,
    out: Array | None = None,
    C: Array | None = None,
    work: Array | None = None,
    both: bool = False,
):
    """Evaluate the PKD Vandermonde gradient on the biunit simplex."""
    x = np.asarray(x)
    N = npolys(d, n)
    if out is not None:
        Vg = out
    elif C is None:
        Vg = np.empty(x.shape[:-1] + (N, d))
    else:
        Vg = np.empty(x.shape[:-1] + (C.shape[1], d))

    if both:
        if C is None:
            V = np.empty(x.shape[:-1] + (N,))
        else:
            V = np.empty(x.shape[:-1] + (C.shape[1],))
        V[:] = 0.0
    Vg[:] = 0.0

    if work is None:
        work = np.empty(x.shape[:-1] + (d,))
    for k, i in enumerate(multiindex_up_to(d, n)):
        if both:
            v, work = proriolkoornwinderdubinergrad(d, i, x, out=work, both=True)
        else:
            work = proriolkoornwinderdubinergrad(d, i, x, out=work)
        if C is None:
            if both:
                V[:, k] = v
            Vg[:, k, :d] = work
        else:
            if both:
                V[:, :] += np.einsum("i,j->ij", v, C[k, :])
            Vg[:, :, :] += np.einsum("ij,k->ikj", work, C[k, :])

    if both:
        return V, Vg
    return Vg
