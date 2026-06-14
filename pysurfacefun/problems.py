"""Problem wrappers for surface PDE solves."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from numbers import Number
from typing import Any, Callable

import numpy as np

from .core import SurfaceFunction, SurfaceMesh, SurfaceOp, surfaceop
from .tri import TriangleSurfaceFunction, TriangleSurfaceMesh, TriangleSurfaceOp, tri_surfaceop


SurfaceScalar = SurfaceFunction | TriangleSurfaceFunction
SurfaceDomain = SurfaceMesh | TriangleSurfaceMesh


def _is_zero(value: Any) -> bool:
    return value is None or (np.isscalar(value) and value == 0)


def _eval_value(value: Any, x, y, z):
    if callable(value):
        return value(x, y, z)
    return value


def _neg_value(value: Any):
    if _is_zero(value):
        return 0.0
    if callable(value):
        return lambda x, y, z: -value(x, y, z)
    return -value


def _add_values(left: Any, right: Any):
    if _is_zero(left):
        return right
    if _is_zero(right):
        return left
    if callable(left) or callable(right):
        return lambda x, y, z: _eval_value(left, x, y, z) + _eval_value(right, x, y, z)
    return left + right


def _mul_value(value: Any, scale: Number):
    if _is_zero(value):
        return 0.0
    if callable(value):
        return lambda x, y, z: scale * value(x, y, z)
    return scale * value


def _copy_surface_scalar(value: SurfaceScalar) -> SurfaceScalar:
    return value.copy()


def _surface_scalar(domain: SurfaceDomain, value: SurfaceScalar | Callable | float) -> SurfaceScalar:
    if isinstance(value, (SurfaceFunction, TriangleSurfaceFunction)):
        if value.domain is not domain:
            raise ValueError("initial field must live on the supplied domain")
        return value.copy()
    if isinstance(domain, TriangleSurfaceMesh):
        from .tri import tri_surfacefun

        return tri_surfacefun(value, domain)
    if isinstance(domain, SurfaceMesh):
        from .core import surfacefun

        return surfacefun(value, domain)
    raise TypeError("expected a SurfaceMesh or TriangleSurfaceMesh domain")


def _call_with_state_time(func: Callable, state: SurfaceScalar, t: float) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(state, t)

    params = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
        return func(state, t)

    positional = [
        param
        for param in params
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return func(state, t)
    if len(positional) == 1:
        return func(state)
    return func()


def _canonical_linear_operator(
    operator: dict[str, Number] | None,
    diffusion: Number,
    linear: Number,
) -> dict[str, Number]:
    out: dict[str, Number] = {}
    if operator is not None:
        for name, value in operator.items():
            key = "b" if name == "c" else name
            out[key] = out.get(key, 0.0) + value
    if diffusion != 0:
        out["lap"] = out.get("lap", 0.0) + diffusion
    if linear != 0:
        out["b"] = out.get("b", 0.0) + linear
    return out


def _implicit_operator(linear_operator: dict[str, Number], dt: float) -> dict[str, Number]:
    op: dict[str, Number] = {"b": 1.0}
    for name, value in linear_operator.items():
        key = "b" if name == "c" else name
        op[key] = op.get(key, 0.0) - dt * value
    return op


@dataclass
class _LinearExpression:
    """Linear expression in one unknown surface field."""

    variable: str | None = None
    terms: dict[str, Number] | None = None
    constant: Any = 0.0

    __array_priority__ = 1000

    def __post_init__(self) -> None:
        if self.terms is None:
            self.terms = {}

    @staticmethod
    def unknown(name: str) -> "_LinearExpression":
        return _LinearExpression(name, {"b": 1.0}, 0.0)

    @staticmethod
    def constant_value(value: Any) -> "_LinearExpression":
        return _LinearExpression(None, {}, value)

    def _merge_variable(self, other: "_LinearExpression") -> str | None:
        if self.variable is None:
            return other.variable
        if other.variable is None or other.variable == self.variable:
            return self.variable
        raise ValueError("SurfaceLBVP currently supports one unknown variable per equation")

    def _combine(self, other: Any, sign: Number) -> "_LinearExpression":
        other = _as_expression(other)
        variable = self._merge_variable(other)
        terms = dict(self.terms)
        for name, value in other.terms.items():
            terms[name] = terms.get(name, 0.0) + sign * value
            if terms[name] == 0:
                del terms[name]
        constant = _add_values(self.constant, _mul_value(other.constant, sign))
        return _LinearExpression(variable, terms, constant)

    def __add__(self, other: Any) -> "_LinearExpression":
        return self._combine(other, 1.0)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_LinearExpression":
        return self._combine(other, -1.0)

    def __rsub__(self, other: Any) -> "_LinearExpression":
        return _as_expression(other)._combine(self, -1.0)

    def __neg__(self) -> "_LinearExpression":
        return _LinearExpression(
            self.variable,
            {name: -value for name, value in self.terms.items()},
            _neg_value(self.constant),
        )

    def __mul__(self, other: Any) -> "_LinearExpression":
        if isinstance(other, _LinearExpression):
            raise ValueError("SurfaceLBVP equations must be linear in the unknown field")
        if not isinstance(other, Number):
            raise ValueError("SurfaceLBVP operator coefficients must be scalar constants")
        return _LinearExpression(
            self.variable,
            {name: other * value for name, value in self.terms.items()},
            _mul_value(self.constant, other),
        )

    def __rmul__(self, other: Any) -> "_LinearExpression":
        return self * other

    def __truediv__(self, other: Any) -> "_LinearExpression":
        if not isinstance(other, Number):
            raise ValueError("SurfaceLBVP equations only support division by scalar constants")
        return self * (1.0 / other)

    def differential(self, name: str) -> "_LinearExpression":
        if self.constant and not _is_zero(self.constant):
            raise ValueError("differential operators may only be applied to unknown-field terms")
        if set(self.terms) - {"b"}:
            raise ValueError("nested differential operators are not supported")
        coeff = self.terms.get("b", 0.0)
        return _LinearExpression(self.variable, {name: coeff}, 0.0)


def _as_expression(value: Any) -> _LinearExpression:
    if isinstance(value, _LinearExpression):
        return value
    return _LinearExpression.constant_value(value)


def _lap(value: Any) -> Any:
    if isinstance(value, _LinearExpression):
        return value.differential("lap")
    if isinstance(value, SurfaceFunction):
        from .core import lap

        return lap(value)
    if isinstance(value, TriangleSurfaceFunction):
        from .tri import tri_lap

        return tri_lap(value)
    raise TypeError("lap expects an unknown field or sampled surface function")


def _diff(name: str) -> Callable[[Any], Any]:
    def apply(value: Any) -> Any:
        if isinstance(value, _LinearExpression):
            return value.differential(name)
        if isinstance(value, SurfaceFunction):
            from .core import diff

            dim = {"dx": 1, "dy": 2, "dz": 3}[name]
            return diff(value, dim=dim)
        raise TypeError("differentiating sampled triangular fields is not supported in SurfaceLBVP equations")

    return apply


@dataclass
class SurfaceEquation:
    """A parsed scalar surface equation."""

    variable: str
    op: dict[str, Number]
    rhs: Any
    source: str | tuple[dict[str, Number], Any]


class SurfaceLBVPSolver:
    """Solver wrapper produced by :class:`SurfaceLBVP`."""

    def __init__(self, problem: "SurfaceLBVP", equation: SurfaceEquation, rankdef: bool = False):
        self.problem = problem
        self.equation = equation
        self.rankdef = rankdef
        self.operator = self._build_operator(equation)
        self.operator.rankdef = rankdef

    def _build_operator(self, equation: SurfaceEquation) -> SurfaceOp | TriangleSurfaceOp:
        if isinstance(self.problem.domain, TriangleSurfaceMesh):
            return tri_surfaceop(self.problem.domain, equation.op, equation.rhs)
        if isinstance(self.problem.domain, SurfaceMesh):
            return surfaceop(self.problem.domain, equation.op, equation.rhs)
        raise TypeError("SurfaceLBVP expects a SurfaceMesh or TriangleSurfaceMesh domain")

    def solve(self) -> SurfaceScalar:
        """Solve the assembled linear boundary-value problem."""
        return self.operator.solve()

    def update_rhs(self, rhs: SurfaceScalar | Callable | float) -> "SurfaceLBVPSolver":
        """Reuse the assembled operator with a new right-hand side."""
        self.operator.update_rhs(rhs)
        self.equation.rhs = rhs
        return self

    def apply(self, rhs: SurfaceScalar | Callable | float) -> SurfaceScalar:
        """Update the right-hand side and immediately solve."""
        self.update_rhs(rhs)
        return self.solve()


class SurfaceLBVP:
    """
    Scalar linear boundary-value problem on a surface mesh.

    The initial API supports one unknown scalar field and equations that are
    linear combinations of ``u``, ``lap(u)``, ``dx(u)``, ``dy(u)``, and ``dz(u)``.
    """

    def __init__(
        self,
        domain: SurfaceDomain,
        variables: str | tuple[str, ...] | list[str] | None = None,
        namespace: dict[str, Any] | None = None,
        rankdef: bool = False,
    ):
        self.domain = domain
        self.namespace = {} if namespace is None else dict(namespace)
        self.variables: dict[str, _LinearExpression] = {}
        self.equations: list[SurfaceEquation] = []
        self.rankdef = rankdef
        if variables is not None:
            if isinstance(variables, str):
                variables = (variables,)
            for name in variables:
                self.field(name)

    def field(self, name: str) -> _LinearExpression:
        """Declare and return an unknown scalar field placeholder."""
        if name in self.variables:
            return self.variables[name]
        if self.variables:
            raise ValueError("SurfaceLBVP currently supports one unknown scalar field")
        field = _LinearExpression.unknown(name)
        self.variables[name] = field
        self.namespace[name] = field
        return field

    def add_equation(
        self,
        equation: str | dict[str, Number] | tuple[dict[str, Number], SurfaceScalar | Callable | float],
        rhs: SurfaceScalar | Callable | float | None = None,
    ) -> SurfaceEquation:
        """Add one linear surface equation to the problem."""
        if self.equations:
            raise ValueError("SurfaceLBVP currently supports a single scalar equation")

        if isinstance(equation, tuple):
            op, eq_rhs = equation
            parsed = self._equation_from_op_rhs(op, eq_rhs)
        elif isinstance(equation, dict):
            if rhs is None:
                raise ValueError("rhs is required when adding an operator dictionary")
            parsed = self._equation_from_op_rhs(equation, rhs)
        elif isinstance(equation, str):
            parsed = self._parse_equation(equation)
        else:
            raise TypeError("equation must be a string, operator dictionary, or (op, rhs) tuple")

        self.equations.append(parsed)
        return parsed

    def _equation_from_op_rhs(self, op: dict[str, Number], rhs: SurfaceScalar | Callable | float) -> SurfaceEquation:
        if not self.variables:
            self.field("u")
        variable = next(iter(self.variables))
        return SurfaceEquation(variable, dict(op), rhs, (dict(op), rhs))

    def _parse_equation(self, equation: str) -> SurfaceEquation:
        if not self.variables:
            self.field("u")
        if equation.count("=") != 1:
            raise ValueError("equations must contain exactly one '='")
        lhs_text, rhs_text = (part.strip() for part in equation.split("=", 1))
        namespace = self._parse_namespace()
        lhs = _as_expression(eval(lhs_text, {"__builtins__": {}}, namespace))
        rhs = _as_expression(eval(rhs_text, {"__builtins__": {}}, namespace))
        residual = lhs - rhs
        if residual.variable is None or not residual.terms:
            raise ValueError("equation must contain the unknown field")
        return SurfaceEquation(
            residual.variable,
            dict(residual.terms),
            _neg_value(residual.constant),
            equation,
        )

    def _parse_namespace(self) -> dict[str, Any]:
        namespace = {
            "lap": _lap,
            "dx": _diff("dx"),
            "dy": _diff("dy"),
            "dz": _diff("dz"),
            "diffx": _diff("dx"),
            "diffy": _diff("dy"),
            "diffz": _diff("dz"),
            "np": np,
        }
        namespace.update(self.namespace)
        namespace.update(self.variables)
        return namespace

    def build_solver(self, rankdef: bool | None = None) -> SurfaceLBVPSolver:
        """Build a reusable solver for the problem."""
        if not self.equations:
            raise ValueError("add an equation before building a solver")
        if rankdef is None:
            rankdef = self.rankdef
        return SurfaceLBVPSolver(self, self.equations[0], rankdef=rankdef)

    def solve(self, rankdef: bool | None = None) -> SurfaceScalar:
        """Build a solver and return its solution."""
        return self.build_solver(rankdef=rankdef).solve()


class SurfaceIVPSolver:
    """Reusable implicit time-stepper produced by :class:`SurfaceIVP`."""

    def __init__(self, problem: "SurfaceIVP", dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.problem = problem
        self.dt = float(dt)
        self.t = float(problem.time)
        self.iteration = 0
        self.state = _copy_surface_scalar(problem.initial)
        self.operator = self._build_operator()
        self.operator.rankdef = problem.rankdef

    def _build_operator(self) -> SurfaceOp | TriangleSurfaceOp:
        op = _implicit_operator(self.problem.linear_operator, self.dt)
        if isinstance(self.problem.domain, TriangleSurfaceMesh):
            return tri_surfaceop(self.problem.domain, op, 0.0)
        if isinstance(self.problem.domain, SurfaceMesh):
            return surfaceop(self.problem.domain, op, 0.0)
        raise TypeError("SurfaceIVP expects a SurfaceMesh or TriangleSurfaceMesh domain")

    def build(self) -> "SurfaceIVPSolver":
        """Assemble the reusable implicit operator."""
        self.operator.build()
        return self

    def rhs(self) -> SurfaceScalar:
        """Return the explicit right-hand side for the next time step."""
        return self.problem.step_rhs(self.state, self.t, self.dt)

    def step(self, dt: float | None = None, rhs: SurfaceScalar | None = None) -> SurfaceScalar:
        """Advance one implicit Euler step and return the new state."""
        if dt is not None:
            if dt <= 0:
                raise ValueError("dt must be positive")
            if float(dt) != self.dt:
                self.dt = float(dt)
                self.operator = self._build_operator()
                self.operator.rankdef = self.problem.rankdef

        step_rhs = self.rhs() if rhs is None else rhs
        self.state = self.operator.apply(step_rhs)
        self.t += self.dt
        self.iteration += 1
        return self.state

    def evaluator_state(self) -> dict[str, Any]:
        """Return a state dictionary suitable for evaluator callbacks."""
        return {
            "u": self.state,
            "state": self.state,
            "t": self.t,
            "iteration": self.iteration,
            "solver": self,
        }

    def run(self, steps: int, evaluator: Any | None = None, evaluate_initial: bool = False) -> SurfaceScalar:
        """Advance ``steps`` times, optionally evaluating outputs each step."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if evaluator is not None and evaluate_initial:
            evaluator.evaluate(self.iteration, time=self.t, state=self.evaluator_state())
        for _ in range(int(steps)):
            self.step()
            if evaluator is not None:
                evaluator.evaluate(self.iteration, time=self.t, state=self.evaluator_state())
        return self.state


class SurfaceIVP:
    """
    Scalar surface initial-value problem with an implicit linear part.

    The equation is advanced as ``u_t = L(u) + reaction(u, t)`` using
    first-order implicit Euler for ``L`` and explicit Euler for ``reaction``.
    """

    def __init__(
        self,
        domain_or_initial: SurfaceDomain | SurfaceScalar,
        initial: SurfaceScalar | Callable | float | None = None,
        *,
        operator: dict[str, Number] | None = None,
        diffusion: Number = 0.0,
        linear: Number = 0.0,
        reaction: Callable[[SurfaceScalar, float], SurfaceScalar | float] | None = None,
        time: float = 0.0,
        rankdef: bool = False,
    ):
        if isinstance(domain_or_initial, (SurfaceFunction, TriangleSurfaceFunction)):
            if initial is not None:
                raise ValueError("pass either an initial field or a domain with initial data, not both")
            self.domain = domain_or_initial.domain
            self.initial = domain_or_initial.copy()
        elif isinstance(domain_or_initial, (SurfaceMesh, TriangleSurfaceMesh)):
            self.domain = domain_or_initial
            init = 0.0 if initial is None else initial
            self.initial = _surface_scalar(self.domain, init)
        else:
            raise TypeError("SurfaceIVP expects an initial field or a surface mesh")

        self.linear_operator = _canonical_linear_operator(operator, diffusion, linear)
        self.reaction = reaction
        self.time = float(time)
        self.rankdef = rankdef

    def implicit_operator(self, dt: float) -> dict[str, Number]:
        """Return the operator used for one implicit Euler step."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        return _implicit_operator(self.linear_operator, float(dt))

    def step_rhs(self, state: SurfaceScalar, t: float, dt: float) -> SurfaceScalar:
        """Return ``state + dt * reaction(state, t)`` for one IMEX step."""
        rhs = state
        if self.reaction is not None:
            rhs = rhs + dt * _call_with_state_time(self.reaction, state, t)
        return rhs

    def build_solver(self, dt: float) -> SurfaceIVPSolver:
        """Build a reusable time-stepper for fixed-size implicit steps."""
        return SurfaceIVPSolver(self, dt)


class SurfaceProblem(SurfaceLBVP):
    """Alias for :class:`SurfaceLBVP` for the current scalar LBVP API."""
