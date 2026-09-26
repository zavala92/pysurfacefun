"""
Implicit-explicit (IMEX) time steppers for ``u_t = L u + N(u, t)``.

``L`` is a linear surface operator treated implicitly and ``N`` an arbitrary
(nonlinear, coupled) term treated explicitly.  Every implicit stage of every
scheme solves

.. math::

    (I - \\gamma\\,\\Delta t\\,L)\\,u = R

with a scheme-dependent constant :math:`\\gamma`, so the HPS factorization of
that operator is computed once and reused for all steps (the multistep
schemes factor one extra operator for their Runge--Kutta start-up steps).

Available schemes
-----------------
======== ===== ===========================================================
name     order description
======== ===== ===========================================================
sbdf1    1     IMEX Euler (backward/forward Euler)
sbdf2    2     semi-implicit BDF2 (started with rk443)
sbdf3    3     semi-implicit BDF3 (started with rk443)
sbdf4    4     semi-implicit BDF4 (started with rk443)
cnab1    1     Crank--Nicolson / forward Euler
cnab2    2     Crank--Nicolson / Adams--Bashforth 2
rk111    1     IMEX Euler written as a Runge--Kutta scheme
rk222    2     Ascher--Ruuth--Spiteri (2,2,2), L-stable implicit part
rk443    3     Ascher--Ruuth--Spiteri (4,4,3), L-stable implicit part
======== ===== ===========================================================

The Runge--Kutta schemes are stiffly accurate: the last stage is the new
solution.  Their implicit stages need ``L U_j`` of earlier stages, which is
recovered from the stage solve as ``(U_j - R_j) / (gamma dt)``.  This is exact
at the collocation (interior) nodes -- the only nodes at which the direct
solver reads a right-hand side -- so no explicit differentiation is needed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

State = dict[str, Any]


class StepContext(Protocol):
    """What a scheme needs from the solver driving it."""

    def solve(self, gamma: float, rhs: State) -> State:
        """Solve ``(I - gamma dt L) u = rhs`` for every variable."""

    def explicit(self, state: State, t: float) -> State:
        """Explicit term ``N(state, t)`` for every variable."""

    def apply_linear(self, state: State) -> State:
        """``L state`` evaluated by spectral differentiation."""


def _combine(terms: list[tuple[float, State]]) -> State:
    """Linear combination ``sum c_i * state_i`` (skipping zero coefficients)."""
    out: State = {}
    for coeff, state in terms:
        if coeff == 0:
            continue
        for name, value in state.items():
            out[name] = coeff * value if name not in out else out[name] + coeff * value
    return out


@dataclass
class _History:
    states: deque = field(default_factory=deque)
    explicit: deque = field(default_factory=deque)
    linear: State | None = None


class Scheme:
    """Base class of IMEX schemes."""

    name = "scheme"
    order = 1
    gamma = 1.0

    def new_history(self) -> _History:
        return _History()

    def gammas(self) -> list[float]:
        """Implicit coefficients used during start-up and in steady operation."""
        return [self.gamma]

    def step(self, ctx: StepContext, state: State, t: float, dt: float, history: _History) -> State:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, order={self.order})"


# Semi-implicit BDF coefficients: a[0] u^{n+1} + sum_j a[j] u^{n+1-j} = dt (L u^{n+1} + sum_j b[j-1] N^{n+1-j}).
_SBDF_COEFFICIENTS = {
    1: ([1.0, -1.0], [1.0]),
    2: ([1.5, -2.0, 0.5], [2.0, -1.0]),
    3: ([11.0 / 6.0, -3.0, 1.5, -1.0 / 3.0], [3.0, -3.0, 1.0]),
    4: ([25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 0.25], [4.0, -6.0, 4.0, -1.0]),
}


class SBDF(Scheme):
    """Semi-implicit backward differentiation formula of order ``k`` (1 to 4).

    The ``k - 1`` start-up steps of the multistep formula are taken with the
    third-order self-starting :func:`rk443 <get_scheme>` scheme, so the global
    error is ``O(dt**k)`` (a low-order ramp would cap it at ``O(dt**2)``).
    Changing ``dt`` restarts the history.
    """

    def __init__(self, order: int):
        if order not in _SBDF_COEFFICIENTS:
            raise ValueError("SBDF order must be 1, 2, 3, or 4")
        self.order = order
        self.name = f"sbdf{order}"
        self.gamma = 1.0 / _SBDF_COEFFICIENTS[order][0][0]
        self._startup = _ars443() if order > 1 else None

    def gammas(self) -> list[float]:
        return [self.gamma] if self._startup is None else [self._startup.gamma, self.gamma]

    def step(self, ctx: StepContext, state: State, t: float, dt: float, history: _History) -> State:
        explicit = ctx.explicit(state, t)
        history.states.appendleft(state)
        history.explicit.appendleft(explicit)
        while len(history.states) > self.order:
            history.states.pop()
            history.explicit.pop()
        q = len(history.states)
        if q < self.order:
            assert self._startup is not None
            return self._startup.step(ctx, state, t, dt, _History(), explicit0=explicit)
        a, b = _SBDF_COEFFICIENTS[q]
        terms = [(-a[j] / a[0], history.states[j - 1]) for j in range(1, q + 1)]
        terms += [(dt * b[j - 1] / a[0], history.explicit[j - 1]) for j in range(1, q + 1)]
        return ctx.solve(1.0 / a[0], _combine(terms))


class CNAB(Scheme):
    """Crank--Nicolson for ``L`` with forward Euler (``order=1``) or Adams--Bashforth 2 for ``N``."""

    gamma = 0.5

    def __init__(self, order: int):
        if order not in (1, 2):
            raise ValueError("CNAB order must be 1 or 2")
        self.order = order
        self.name = f"cnab{order}"

    def step(self, ctx: StepContext, state: State, t: float, dt: float, history: _History) -> State:
        explicit = ctx.explicit(state, t)
        linear = history.linear if history.linear is not None else ctx.apply_linear(state)
        if self.order == 2 and history.explicit:
            nterm = _combine([(1.5, explicit), (-0.5, history.explicit[0])])
        else:
            nterm = explicit
        rhs = _combine([(1.0, state), (0.5 * dt, linear), (dt, nterm)])
        new = ctx.solve(self.gamma, rhs)
        # L u^{n+1} at the collocation nodes, recovered from the solve.
        history.linear = _combine([(1.0 / (self.gamma * dt), new), (-1.0 / (self.gamma * dt), rhs)])
        history.explicit.clear()
        history.explicit.append(explicit)
        return new


class IMEXRungeKutta(Scheme):
    """Stiffly accurate IMEX Runge--Kutta scheme of Ascher--Ruuth--Spiteri type.

    ``A`` (implicit) and ``Ahat`` (explicit) are ``(s + 1) x (s + 1)`` tableaux
    whose first stage is explicit (``A[:, 0] = 0``) and whose implicit
    diagonal is the constant ``gamma``.
    """

    def __init__(self, name: str, order: int, A: np.ndarray, Ahat: np.ndarray):
        A = np.asarray(A, dtype=float)
        Ahat = np.asarray(Ahat, dtype=float)
        diag = np.diag(A)[1:]
        if not np.allclose(diag, diag[0]) or np.any(A[:, 0] != 0):
            raise ValueError("expected an ARS-type tableau with a constant implicit diagonal")
        self.name = name
        self.order = order
        self.A = A
        self.Ahat = Ahat
        self.c = A.sum(axis=1)
        self.chat = Ahat.sum(axis=1)
        self.gamma = float(diag[0])

    def step(
        self, ctx: StepContext, state: State, t: float, dt: float, history: _History, explicit0: State | None = None
    ) -> State:
        A, Ahat = self.A, self.Ahat
        nstages = A.shape[0]
        linear: list[State | None] = [None]
        explicit: list[State] = [ctx.explicit(state, t + self.chat[0] * dt) if explicit0 is None else explicit0]
        stage = state
        for i in range(1, nstages):
            terms: list[tuple[float, State]] = [(1.0, state)]
            for j in range(i):
                if A[i, j] != 0 and linear[j] is not None:
                    terms.append((dt * A[i, j], linear[j]))  # type: ignore[arg-type]
                if Ahat[i, j] != 0:
                    terms.append((dt * Ahat[i, j], explicit[j]))
            rhs = _combine(terms)
            stage = ctx.solve(self.gamma, rhs)
            if i < nstages - 1:
                scale = 1.0 / (self.gamma * dt)
                linear.append(_combine([(scale, stage), (-scale, rhs)]))
                explicit.append(ctx.explicit(stage, t + self.chat[i] * dt))
        return stage


def _ars222() -> IMEXRungeKutta:
    gamma = 1.0 - 1.0 / np.sqrt(2.0)
    delta = 1.0 - 1.0 / (2.0 * gamma)
    A = [[0, 0, 0], [0, gamma, 0], [0, 1 - gamma, gamma]]
    Ahat = [[0, 0, 0], [gamma, 0, 0], [delta, 1 - delta, 0]]
    return IMEXRungeKutta("rk222", 2, np.array(A), np.array(Ahat))


def _ars443() -> IMEXRungeKutta:
    A = [
        [0, 0, 0, 0, 0],
        [0, 1 / 2, 0, 0, 0],
        [0, 1 / 6, 1 / 2, 0, 0],
        [0, -1 / 2, 1 / 2, 1 / 2, 0],
        [0, 3 / 2, -3 / 2, 1 / 2, 1 / 2],
    ]
    Ahat = [
        [0, 0, 0, 0, 0],
        [1 / 2, 0, 0, 0, 0],
        [11 / 18, 1 / 18, 0, 0, 0],
        [5 / 6, -5 / 6, 1 / 2, 0, 0],
        [1 / 4, 7 / 4, 3 / 4, -7 / 4, 0],
    ]
    return IMEXRungeKutta("rk443", 3, np.array(A), np.array(Ahat))


def _ars111() -> IMEXRungeKutta:
    return IMEXRungeKutta("rk111", 1, np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 0]]))


_FACTORIES: dict[str, Callable[[], Scheme]] = {
    "sbdf1": lambda: SBDF(1),
    "sbdf2": lambda: SBDF(2),
    "sbdf3": lambda: SBDF(3),
    "sbdf4": lambda: SBDF(4),
    "cnab1": lambda: CNAB(1),
    "cnab2": lambda: CNAB(2),
    "rk111": _ars111,
    "rk222": _ars222,
    "rk443": _ars443,
}
_ALIASES = {
    "euler": "sbdf1",
    "imex_euler": "sbdf1",
    "bdf1": "sbdf1",
    "bdf2": "sbdf2",
    "bdf3": "sbdf3",
    "bdf4": "sbdf4",
    "ars222": "rk222",
    "ars443": "rk443",
    "ars111": "rk111",
}

#: Names of the available schemes.
SCHEMES = tuple(_FACTORIES)


def get_scheme(scheme: str | Scheme) -> Scheme:
    """Return a scheme instance from its name (case insensitive) or pass one through."""
    if isinstance(scheme, Scheme):
        return scheme
    key = str(scheme).lower().replace("-", "").replace(" ", "")
    key = _ALIASES.get(key, key)
    if key not in _FACTORIES:
        raise ValueError(f"unknown time-stepping scheme {scheme!r}; choose from {', '.join(SCHEMES)}")
    return _FACTORIES[key]()
