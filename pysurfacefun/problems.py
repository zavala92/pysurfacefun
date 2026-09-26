"""
Boundary-value and initial-value problems on surfaces.

Problems are posed with equation strings in the style of Dedalus::

    problem = psf.SurfaceLBVP(dom, variables="u", namespace={"k": k, "f": f})
    problem.add_equation("-div(k*grad(u)) + u = f")
    u = problem.solve()

    ivp = psf.SurfaceIVP(dom, variables={"u": u0, "v": v0}, namespace=params)
    ivp.add_equation("dt(u) - Du*lap(u) = u - u**3 - v")
    ivp.add_equation("dt(v) - Dv*lap(v) = eps*(u - a*v)")
    solver = ivp.build_solver(dt=0.1, scheme="rk443")
    solver.run(1000)

Left-hand sides are linear in the unknown and are discretized implicitly with
the fast direct solver.  Coefficients may be numbers, callables ``f(x, y, z)``,
or surface fields; derivatives follow the tangential product rule, so
``div(k*grad(u))`` expands to ``k lap(u) + grad(k).grad(u)``.  Right-hand sides
of initial-value problems are arbitrary expressions in the fields and ``t``,
evaluated explicitly.
"""

from __future__ import annotations

import builtins
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np

from . import core, unified
from .core import SurfaceFunction, SurfaceMesh, SurfaceOp
from .fields import PatchField, PatchVectorField
from .operators import HPSOperator, is_zero
from .timesteppers import SBDF, Scheme, get_scheme
from .tri import TriangleSurfaceFunction, TriangleSurfaceMesh, TriangleSurfaceOp

SurfaceScalar = SurfaceFunction | TriangleSurfaceFunction
SurfaceDomain = SurfaceMesh | TriangleSurfaceMesh

_AXES = "xyz"
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in ("abs", "min", "max", "pow", "float", "complex", "int", "sum", "range", "len")
}


# ---------------------------------------------------------------------------
# Scalar helpers.
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, Number) or (isinstance(value, np.ndarray) and value.ndim == 0)


def _is_zero(value: Any) -> bool:
    return is_zero(value)


def _eval_value(value: Any, x, y, z):
    if callable(value) and not isinstance(value, PatchField):
        return value(x, y, z)
    return value


def _neg_value(value: Any):
    if _is_zero(value):
        return 0.0
    if callable(value) and not isinstance(value, PatchField):
        return lambda x, y, z: -value(x, y, z)
    return -value


def _add_values(left: Any, right: Any):
    if _is_zero(left):
        return right
    if _is_zero(right):
        return left
    left_fn = callable(left) and not isinstance(left, PatchField)
    right_fn = callable(right) and not isinstance(right, PatchField)
    if left_fn or right_fn:
        if isinstance(left, PatchField) or isinstance(right, PatchField):
            field_value = left if isinstance(left, PatchField) else right
            fn = right if left_fn is False else left
            return field_value + unified.field(fn, field_value.domain)
        return lambda x, y, z: _eval_value(left, x, y, z) + _eval_value(right, x, y, z)
    return left + right


def _mul_value(value: Any, scale: Any):
    if _is_zero(value) or _is_zero(scale):
        return 0.0
    if callable(value) and not isinstance(value, PatchField):
        if _is_number(scale):
            return lambda x, y, z: scale * value(x, y, z)
        return scale * unified.field(value, scale.domain)
    return scale * value


def _copy_surface_scalar(value: SurfaceScalar) -> SurfaceScalar:
    return value.copy()


def _surface_scalar(domain: SurfaceDomain, value: Any) -> SurfaceScalar:
    if isinstance(value, (SurfaceFunction, TriangleSurfaceFunction)):
        if value.domain is not domain:
            raise ValueError("initial field must live on the supplied domain")
        return value.copy()
    return unified.field(value, domain)


def _positional_arity(func: Callable) -> int | None:
    """Number of positional parameters, or ``None`` for ``*args`` or uninspectable callables."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    params = list(signature.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return None
    return sum(p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in params)


def _call_with_state_time(func: Callable, state: Any, t: float) -> Any:
    arity = _positional_arity(func)
    if arity is None or arity >= 2:
        return func(state, t)
    if arity == 1:
        return func(state)
    return func()


def _canonical_linear_operator(operator: dict[str, Any] | None, diffusion: Any, linear: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if operator is not None:
        for name, value in operator.items():
            key = "b" if name == "c" else name
            out[key] = _add_values(out.get(key, 0.0), value)
    if not _is_zero(diffusion):
        out["lap"] = _add_values(out.get("lap", 0.0), diffusion)
    if not _is_zero(linear):
        out["b"] = _add_values(out.get("b", 0.0), linear)
    return out


def _implicit_operator(linear_operator: dict[str, Any], dt: float) -> dict[str, Any]:
    """Operator of ``(I - dt L)`` for ``L`` given as a coefficient dictionary."""
    op: dict[str, Any] = {"b": 1.0}
    for name, value in linear_operator.items():
        key = "b" if name == "c" else name
        op[key] = _add_values(op.get(key, 0.0), _mul_value(value, -dt))
    return op


# ---------------------------------------------------------------------------
# Linear expressions in one unknown.
# ---------------------------------------------------------------------------

_DT = ("dt",)


class _LinearExpression:
    """Linear expression ``sum_w a_w D^w u + constant`` in one unknown field ``u``.

    ``terms`` maps derivative words to coefficients: ``()`` is ``u`` itself,
    ``(c,)`` is ``D_c u``, ``(c, d)`` is ``D_c D_d u`` (``D_d`` applied first),
    and ``("dt",)`` marks the time derivative in initial-value problems.
    Coefficients are numbers or surface fields; ``constant`` holds data
    (numbers, fields, or callables).
    """

    __array_priority__ = 1000
    __array_ufunc__ = None

    def __init__(self, variable: str | None = None, terms: dict | None = None, constant: Any = 0.0, domain: Any = None):
        self.variable = variable
        self.terms: dict[tuple, Any] = {} if terms is None else dict(terms)
        self.constant = constant
        self.domain = domain

    @staticmethod
    def unknown(name: str, domain: Any = None) -> _LinearExpression:
        return _LinearExpression(name, {(): 1.0}, 0.0, domain)

    @staticmethod
    def constant_value(value: Any, domain: Any = None) -> _LinearExpression:
        return _LinearExpression(None, {}, value, domain)

    def __repr__(self) -> str:
        parts = [f"{_word_name(w)}: {c!r}" for w, c in self.terms.items()]
        return f"_LinearExpression({self.variable!r}, {{{', '.join(parts)}}}, constant={self.constant!r})"

    # -- coefficient helpers -------------------------------------------------

    def _coefficient(self, value: Any) -> Any:
        """Convert a multiplier into a number or a field on the problem domain."""
        if _is_number(value) or isinstance(value, PatchField):
            return value
        if callable(value):
            if self.domain is None:
                raise ValueError("callable coefficients need a problem domain")
            return unified.field(value, self.domain)
        if isinstance(value, np.ndarray) and self.domain is not None:
            return unified.field(value, self.domain) if value.ndim else value.item()
        raise TypeError(f"unsupported coefficient of type {type(value).__name__}")

    def _merge_variable(self, other: _LinearExpression) -> str | None:
        if self.variable is None:
            return other.variable
        if other.variable is None or other.variable == self.variable:
            return self.variable
        raise ValueError("implicit terms may only involve the equation's own unknown")

    # -- arithmetic ----------------------------------------------------------

    def _combine(self, other: Any, sign: float) -> _LinearExpression:
        other = _as_expression(other, self.domain)
        variable = self._merge_variable(other)
        terms = dict(self.terms)
        for word, value in other.terms.items():
            current = terms.get(word, 0.0)
            new = _add_values(current, _mul_value(value, sign) if sign != 1.0 else value)
            if _is_zero(new):
                terms.pop(word, None)
            else:
                terms[word] = new
        constant = _add_values(self.constant, _mul_value(other.constant, sign) if sign != 1.0 else other.constant)
        return _LinearExpression(variable, terms, constant, self.domain or other.domain)

    def __add__(self, other: Any) -> _LinearExpression:
        return self._combine(other, 1.0)

    __radd__ = __add__

    def __sub__(self, other: Any) -> _LinearExpression:
        return self._combine(other, -1.0)

    def __rsub__(self, other: Any) -> _LinearExpression:
        return _as_expression(other, self.domain)._combine(self, -1.0)

    def __neg__(self) -> _LinearExpression:
        return _LinearExpression(
            self.variable,
            {w: _mul_value(c, -1.0) for w, c in self.terms.items()},
            _neg_value(self.constant),
            self.domain,
        )

    def __pos__(self) -> _LinearExpression:
        return self

    def __mul__(self, other: Any) -> _LinearExpression:
        if isinstance(other, (_LinearExpression, _LinearVector)):
            raise ValueError("equations must be linear in the unknown field")
        coeff = self._coefficient(other)
        terms = {w: _mul_value(c, coeff) for w, c in self.terms.items()}
        terms = {w: c for w, c in terms.items() if not _is_zero(c)}
        return _LinearExpression(self.variable, terms, _mul_value(self.constant, coeff), self.domain)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> _LinearExpression:
        if isinstance(other, (_LinearExpression, _LinearVector)):
            raise ValueError("cannot divide by an expression in the unknown field")
        coeff = self._coefficient(other)
        return self * (1.0 / coeff)

    # -- differential operators ----------------------------------------------

    def derivative(self, axis: int) -> _LinearExpression:
        """Tangential derivative ``D_axis`` with the product rule for field coefficients."""
        terms: dict[tuple, Any] = {}

        def add(word: tuple, value: Any) -> None:
            new = _add_values(terms.get(word, 0.0), value)
            if _is_zero(new):
                terms.pop(word, None)
            else:
                terms[word] = new

        for word, coeff in self.terms.items():
            if word == _DT:
                raise ValueError("spatial derivatives of dt(u) are not supported")
            if len(word) >= 2:
                raise ValueError("derivatives of order higher than two are not supported")
            add((axis, *word), coeff)
            if isinstance(coeff, PatchField):
                add(word, unified.diff(coeff, dim=axis + 1))
        constant = self.constant
        if _is_zero(constant) or _is_number(constant):
            dconst: Any = 0.0
        else:
            if callable(constant) and not isinstance(constant, PatchField):
                constant = unified.field(constant, self.domain)
            dconst = unified.diff(constant, dim=axis + 1)
        return _LinearExpression(self.variable, terms, dconst, self.domain)

    def differential(self, name: str) -> _LinearExpression:
        """Apply ``"lap"``, ``"dx"``, ``"dy"``, or ``"dz"`` (legacy helper)."""
        if name == "lap":
            return _lap(self)
        return self.derivative(_AXES.index(name[-1]))

    def to_operator(self) -> dict[str, Any]:
        """Coefficient dictionary accepted by the surface operators."""
        out: dict[str, Any] = {}
        for word, coeff in self.terms.items():
            if word == _DT:
                continue
            out[_word_name(word)] = coeff
        diag = [out.get(name) for name in ("dxx", "dyy", "dzz")]
        if all(_is_number(v) for v in diag) and diag[0] == diag[1] == diag[2]:
            for name in ("dxx", "dyy", "dzz"):
                out.pop(name)
            out = {"lap": diag[0], **out}
        return out


def _word_name(word: tuple) -> str:
    if word == ():
        return "b"
    if word == _DT:
        return "dt"
    return "d" + "".join(_AXES[a] for a in word)


class _LinearVector:
    """Vector of three linear expressions (e.g. ``grad(u)``)."""

    __array_priority__ = 1000
    __array_ufunc__ = None

    def __init__(self, components):
        self.components = tuple(components)

    def __mul__(self, other: Any) -> _LinearVector:
        return _LinearVector(c * other for c in self.components)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> _LinearVector:
        return _LinearVector(c / other for c in self.components)

    def __add__(self, other: Any) -> _LinearVector:
        parts = _vector_parts(other)
        return _LinearVector(a + b for a, b in zip(self.components, parts))

    __radd__ = __add__

    def __sub__(self, other: Any) -> _LinearVector:
        parts = _vector_parts(other)
        return _LinearVector(a - b for a, b in zip(self.components, parts))

    def __rsub__(self, other: Any) -> _LinearVector:
        return (-self) + other

    def __neg__(self) -> _LinearVector:
        return _LinearVector(-c for c in self.components)


def _vector_parts(value: Any) -> tuple:
    if isinstance(value, _LinearVector):
        return value.components
    if isinstance(value, PatchVectorField):
        return value.components
    arr = value if isinstance(value, (list, tuple)) else np.asarray(value)
    if len(arr) != 3:
        raise ValueError("vectors must have three components")
    return tuple(arr)


def _as_expression(value: Any, domain: Any = None) -> _LinearExpression:
    if isinstance(value, _LinearExpression):
        return value
    if isinstance(value, _LinearVector):
        raise ValueError("equations must be scalar; did you forget div(...) or dot(...)?")
    return _LinearExpression.constant_value(value, domain)


# -- namespace functions (symbolic for unknowns, numeric for fields) ---------


def _derivative(axis: int) -> Callable[[Any], Any]:
    def apply(value: Any) -> Any:
        if isinstance(value, _LinearExpression):
            return value.derivative(axis)
        if isinstance(value, PatchField):
            return unified.diff(value, dim=axis + 1)
        raise TypeError("derivatives apply to unknowns or surface fields")

    apply.__name__ = "d" + _AXES[axis]
    return apply


def _lap(value: Any) -> Any:
    if isinstance(value, _LinearExpression):
        out = value.derivative(0).derivative(0)
        out = out + value.derivative(1).derivative(1)
        return out + value.derivative(2).derivative(2)
    if isinstance(value, PatchField):
        return unified.lap(value)
    raise TypeError("lap expects an unknown field or sampled surface function")


def _grad(value: Any) -> Any:
    if isinstance(value, _LinearExpression):
        return _LinearVector(value.derivative(a) for a in range(3))
    if isinstance(value, PatchField):
        return unified.grad(value)
    raise TypeError("grad expects an unknown field or sampled surface function")


def _div(value: Any) -> Any:
    if isinstance(value, _LinearVector) or (
        isinstance(value, (list, tuple)) and any(isinstance(c, _LinearExpression) for c in value)
    ):
        parts = _vector_parts(value)
        out = _as_expression(parts[0]).derivative(0)
        out = out + _as_expression(parts[1], out.domain).derivative(1)
        return out + _as_expression(parts[2], out.domain).derivative(2)
    if isinstance(value, PatchVectorField):
        return unified.div(value)
    raise TypeError("div expects a vector expression or vector field")


def _dot(a: Any, b: Any) -> Any:
    if isinstance(a, _LinearVector) or isinstance(b, _LinearVector):
        pa, pb = _vector_parts(a), _vector_parts(b)
        out = pa[0] * pb[0]
        out = out + pa[1] * pb[1]
        return out + pa[2] * pb[2]
    return core.dot(a, b)


def _expression_functions() -> dict[str, Any]:
    ufuncs = {
        name: getattr(np, name)
        for name in ("sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sinh", "cosh", "arctan")
    }
    return {
        "lap": _lap,
        "laplacian": _lap,
        "grad": _grad,
        "div": _div,
        "dot": _dot,
        "dx": _derivative(0),
        "dy": _derivative(1),
        "dz": _derivative(2),
        "diffx": _derivative(0),
        "diffy": _derivative(1),
        "diffz": _derivative(2),
        "real": np.real,
        "imag": np.imag,
        "conj": np.conj,
        "pi": np.pi,
        "np": np,
        **ufuncs,
    }


_EQUALS = re.compile(r"(?<![<>=!])=(?!=)")


def _split_equation(equation: str) -> tuple[str, str]:
    parts = _EQUALS.split(equation)
    if len(parts) != 2:
        raise ValueError("equations must contain exactly one '='")
    return parts[0].strip(), parts[1].strip()


def _evaluate(text: str, namespace: dict[str, Any]) -> Any:
    return eval(text, {"__builtins__": _SAFE_BUILTINS}, namespace)


# ---------------------------------------------------------------------------
# Linear boundary-value problems.
# ---------------------------------------------------------------------------


@dataclass
class SurfaceEquation:
    """A parsed scalar surface equation ``L(u) = rhs``."""

    variable: str
    op: dict[str, Any]
    rhs: Any
    source: str | tuple[dict[str, Any], Any]


def _make_operator(
    domain: SurfaceDomain, op: dict[str, Any], rhs: Any = 0.0, **kwargs
) -> SurfaceOp | TriangleSurfaceOp:
    if isinstance(domain, TriangleSurfaceMesh):
        return TriangleSurfaceOp(domain, op, rhs, **kwargs)
    if isinstance(domain, SurfaceMesh):
        return SurfaceOp(domain, op, rhs, **kwargs)
    raise TypeError("expected a SurfaceMesh or TriangleSurfaceMesh domain")


class SurfaceLBVPSolver:
    """Reusable solver produced by :class:`SurfaceLBVP`."""

    def __init__(self, problem: SurfaceLBVP, equation: SurfaceEquation, rankdef: bool = False, **operator_kwargs):
        self.problem = problem
        self.equation = equation
        self.rankdef = rankdef
        self.operator = _make_operator(problem.domain, equation.op, equation.rhs, rankdef=rankdef, **operator_kwargs)

    def solve(self) -> SurfaceScalar:
        """Solve the assembled linear boundary-value problem."""
        return self.operator.solve()

    def update_rhs(self, rhs: Any) -> SurfaceLBVPSolver:
        """Reuse the factorization with a new right-hand side."""
        self.operator.update_rhs(rhs)
        self.equation.rhs = rhs
        return self

    def apply(self, rhs: Any) -> SurfaceScalar:
        """Update the right-hand side and immediately solve."""
        self.update_rhs(rhs)
        return self.solve()


class SurfaceLBVP:
    """Scalar linear boundary-value problem on a surface mesh.

    Equations are linear in one unknown and may use ``lap``, ``grad``,
    ``div``, ``dot``, ``dx``, ``dy``, ``dz`` with constant or variable
    coefficients, for example ``"-div(k*grad(u)) + c*u = f"``.
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
        """Declare and return the unknown scalar field."""
        if name in self.variables:
            return self.variables[name]
        if self.variables:
            raise ValueError("SurfaceLBVP supports one unknown scalar field")
        unknown = _LinearExpression.unknown(name, self.domain)
        self.variables[name] = unknown
        self.namespace[name] = unknown
        return unknown

    def add_equation(self, equation: str | dict | tuple, rhs: Any = None) -> SurfaceEquation:
        """Add the equation: a string, an operator dictionary with ``rhs``, or an ``(op, rhs)`` tuple."""
        if self.equations:
            raise ValueError("SurfaceLBVP supports a single scalar equation")
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

    def _equation_from_op_rhs(self, op: dict[str, Any], rhs: Any) -> SurfaceEquation:
        if not self.variables:
            self.field("u")
        variable = next(iter(self.variables))
        return SurfaceEquation(variable, dict(op), rhs, (dict(op), rhs))

    def _parse_equation(self, equation: str) -> SurfaceEquation:
        if not self.variables:
            self.field("u")
        lhs_text, rhs_text = _split_equation(equation)
        namespace = self._parse_namespace()
        lhs = _as_expression(_evaluate(lhs_text, namespace), self.domain)
        rhs = _as_expression(_evaluate(rhs_text, namespace), self.domain)
        residual = lhs - rhs
        if residual.variable is None or not residual.terms:
            raise ValueError("equation must contain the unknown field")
        if _DT in residual.terms:
            raise ValueError("boundary-value problems cannot contain dt(...); use SurfaceIVP")
        return SurfaceEquation(residual.variable, residual.to_operator(), _neg_value(residual.constant), equation)

    def _parse_namespace(self) -> dict[str, Any]:
        namespace = _expression_functions()
        namespace.update(self.namespace)
        namespace.update(self.variables)
        return namespace

    def build_solver(self, rankdef: bool | None = None, **operator_kwargs) -> SurfaceLBVPSolver:
        """Assemble a reusable solver (``operator_kwargs`` go to the surface operator)."""
        if not self.equations:
            raise ValueError("add an equation before building a solver")
        if rankdef is None:
            rankdef = self.rankdef
        return SurfaceLBVPSolver(self, self.equations[0], rankdef=rankdef, **operator_kwargs)

    def solve(self, rankdef: bool | None = None) -> SurfaceScalar:
        """Build a solver and return the solution."""
        return self.build_solver(rankdef=rankdef).solve()


class SurfaceProblem(SurfaceLBVP):
    """Alias of :class:`SurfaceLBVP`."""


# ---------------------------------------------------------------------------
# Initial-value problems.
# ---------------------------------------------------------------------------


@dataclass
class _IVPEquation:
    variable: str
    linear: dict[str, Any]
    rhs_code: Any
    rhs_text: str
    scale: Any


class SurfaceIVP:
    """Surface initial-value problem ``du/dt = L u + N(u, t)`` for one or more fields.

    ``L`` is linear (constant or variable coefficients, per field) and is
    advanced implicitly; ``N`` couples the fields and is advanced explicitly.

    Single field (compact form)::

        ivp = SurfaceIVP(u0, diffusion=1e-3, reaction=lambda u, t: u - u**3)

    Several fields with equations::

        ivp = SurfaceIVP(dom, variables={"u": u0, "v": v0}, namespace={...})
        ivp.add_equation("dt(u) - Du*lap(u) = f(u, v)")
        ivp.add_equation("dt(v) - Dv*lap(v) = g(u, v)")

    or with callables: ``diffusion={"u": Du, "v": Dv}`` and
    ``reaction=lambda u, v, t: (f(u, v), g(u, v))``.
    """

    def __init__(
        self,
        domain_or_initial: SurfaceDomain | SurfaceScalar | None = None,
        initial: Any = None,
        *,
        operator: dict[str, Any] | None = None,
        diffusion: Any = 0.0,
        linear: Any = 0.0,
        reaction: Callable | None = None,
        time: float = 0.0,
        rankdef: bool = False,
        variables: dict[str, Any] | list[str] | tuple[str, ...] | str | None = None,
        namespace: dict[str, Any] | None = None,
    ):
        self.time = float(time)
        self.rankdef = rankdef
        self.namespace = {} if namespace is None else dict(namespace)
        self.reaction = reaction
        self.equations: dict[str, _IVPEquation] = {}

        if variables is None:
            if isinstance(domain_or_initial, (SurfaceFunction, TriangleSurfaceFunction)):
                if initial is not None:
                    raise ValueError("pass either an initial field or a domain with initial data, not both")
                self.domain = domain_or_initial.domain
                self.initial_state = {"u": domain_or_initial.copy()}
            elif isinstance(domain_or_initial, (SurfaceMesh, TriangleSurfaceMesh)):
                self.domain = domain_or_initial
                self.initial_state = {"u": _surface_scalar(self.domain, 0.0 if initial is None else initial)}
            else:
                raise TypeError("SurfaceIVP expects an initial field or a surface mesh")
        else:
            if isinstance(variables, str):
                variables = [variables]
            if not isinstance(variables, dict):
                variables = dict.fromkeys(variables, 0.0)
            if isinstance(domain_or_initial, (SurfaceMesh, TriangleSurfaceMesh)):
                self.domain = domain_or_initial
            else:
                first = next((v for v in variables.values() if isinstance(v, PatchField)), None)
                if first is None:
                    raise TypeError("pass the surface mesh when no initial field is given")
                self.domain = first.domain
            self.initial_state = {name: _surface_scalar(self.domain, value) for name, value in variables.items()}
            if not self.initial_state:
                raise ValueError("declare at least one variable")

        self.variable_names = list(self.initial_state)
        self.linear_operators = {
            name: _canonical_linear_operator(
                _per_variable(operator, name, dict_valued=True),
                _per_variable(diffusion, name),
                _per_variable(linear, name),
            )
            for name in self.variable_names
        }

    # -- legacy single-field attributes ---------------------------------------

    @property
    def initial(self) -> Any:
        """Initial field (single-field problems) or dictionary of initial fields."""
        if len(self.variable_names) == 1:
            return self.initial_state[self.variable_names[0]]
        return dict(self.initial_state)

    @property
    def linear_operator(self) -> dict[str, Any]:
        """Linear operator of a single-field problem."""
        if len(self.variable_names) != 1:
            raise AttributeError("linear_operator is only defined for single-field problems; see linear_operators")
        return self.linear_operators[self.variable_names[0]]

    # -- equations -----------------------------------------------------------

    def add_equation(self, equation: str) -> _IVPEquation:
        """Add ``"dt(u) + <linear in u> = <explicit expression>"`` for one variable."""
        if self.reaction is not None:
            raise ValueError("use either equations or a reaction callable, not both")
        lhs_text, rhs_text = _split_equation(equation)
        namespace = _expression_functions()
        namespace.update(self.namespace)
        unknowns = {name: _LinearExpression.unknown(name, self.domain) for name in self.variable_names}
        namespace.update(unknowns)

        def dt(value: Any) -> _LinearExpression:
            if not (isinstance(value, _LinearExpression) and set(value.terms) == {()} and value.variable):
                raise ValueError("dt(...) must be applied to a variable")
            return _LinearExpression(value.variable, {_DT: value.terms[()]}, 0.0, self.domain)

        namespace["dt"] = dt
        lhs = _as_expression(_evaluate(lhs_text, namespace), self.domain)
        if lhs.variable is None or _DT not in lhs.terms:
            raise ValueError("the left-hand side must contain dt(<variable>)")
        if not _is_zero(lhs.constant):
            raise ValueError("move data terms to the right-hand side of the equation")
        variable = lhs.variable
        if variable in self.equations:
            raise ValueError(f"variable {variable!r} already has an equation")
        scale = lhs.terms.pop(_DT)
        # dt(u) * scale + M(u) = RHS  =>  u_t = -(M/scale) u + RHS/scale
        linear = (lhs * (-1.0 / scale)).to_operator() if lhs.terms else {}
        code = compile(rhs_text, f"<rhs of {variable}>", "eval")
        eq = _IVPEquation(variable, linear, code, rhs_text, scale)
        self.equations[variable] = eq
        self.linear_operators[variable] = linear
        return eq

    # -- explicit terms ------------------------------------------------------

    def explicit(self, state: dict[str, Any], t: float) -> dict[str, Any]:
        """Explicit right-hand side ``N(state, t)`` of every variable."""
        names = self.variable_names
        if self.equations:
            missing = [name for name in names if name not in self.equations]
            if missing:
                raise ValueError(f"missing equations for {', '.join(missing)}")
            namespace = _expression_functions()
            namespace.update(self.namespace)
            namespace.update(state)
            namespace["t"] = t
            out = {}
            for name in names:
                eq = self.equations[name]
                value = _evaluate(eq.rhs_code, namespace)
                if not _is_number(eq.scale) or eq.scale != 1:
                    value = value / eq.scale
                out[name] = _as_field(value, self.domain, state[name])
            return out
        if self.reaction is None:
            return {name: 0.0 * state[name] for name in names}
        if len(names) == 1:
            value = _call_with_state_time(self.reaction, state[names[0]], t)
            return {names[0]: _as_field(value, self.domain, state[names[0]])}
        args = [state[name] for name in names]
        arity = _positional_arity(self.reaction)
        result = self.reaction(*args, t) if arity is None or arity > len(names) else self.reaction(*args)
        if isinstance(result, dict):
            return {name: _as_field(result[name], self.domain, state[name]) for name in names}
        result = list(result)
        if len(result) != len(names):
            raise ValueError("the reaction callable must return one term per variable")
        return {name: _as_field(value, self.domain, state[name]) for name, value in zip(names, result)}

    def implicit_operator(self, dt: float) -> Any:
        """Operator ``I - dt L`` (a dictionary per variable for multi-field problems)."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        ops = {name: _implicit_operator(self.linear_operators[name], float(dt)) for name in self.variable_names}
        return ops[self.variable_names[0]] if len(ops) == 1 else ops

    def step_rhs(self, state: Any, t: float, dt: float) -> Any:
        """``state + dt * N(state, t)``: the right-hand side of one IMEX Euler step."""
        single = not isinstance(state, dict)
        states = {self.variable_names[0]: state} if single else state
        explicit = self.explicit(states, t)
        rhs = {name: states[name] + dt * explicit[name] for name in self.variable_names}
        return rhs[self.variable_names[0]] if single else rhs

    def build_solver(self, dt: float, scheme: str | Scheme = "sbdf1", **operator_kwargs) -> SurfaceIVPSolver:
        """Reusable time stepper with step size ``dt`` (see :mod:`pysurfacefun.timesteppers`)."""
        return SurfaceIVPSolver(self, dt, scheme=scheme, **operator_kwargs)


def _per_variable(value: Any, name: str, dict_valued: bool = False) -> Any:
    if isinstance(value, dict) and not dict_valued:
        return value.get(name, 0.0)
    if isinstance(value, dict) and dict_valued and value and all(isinstance(v, dict) for v in value.values()):
        return value.get(name)
    return value


def _as_field(value: Any, domain: SurfaceDomain, like: SurfaceScalar) -> SurfaceScalar:
    if isinstance(value, PatchField):
        return value
    if _is_number(value):
        return like._new(np.full(like.data.shape, value, dtype=np.result_type(like.data, value)))
    return unified.field(value, domain)


class SurfaceIVPSolver:
    """Time stepper for a :class:`SurfaceIVP` with a fixed step size.

    Implicit operators ``I - gamma dt L`` are factored lazily, cached, and
    shared between variables with identical constant-coefficient operators
    (which are then solved together as one multi-right-hand-side sweep).
    """

    def __init__(self, problem: SurfaceIVP, dt: float, scheme: str | Scheme = "sbdf1", **operator_kwargs):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.problem = problem
        self.dt = float(dt)
        self.t = float(problem.time)
        self.iteration = 0
        self.scheme = get_scheme(scheme)
        self.names = list(problem.variable_names)
        self.fields = {name: _copy_surface_scalar(problem.initial_state[name]) for name in self.names}
        self._operator_kwargs = operator_kwargs
        self._operators: dict[tuple, HPSOperator] = {}
        self._history = self.scheme.new_history()

    # -- state ---------------------------------------------------------------

    @property
    def state(self) -> Any:
        """Current field (single-field problems) or dictionary of fields."""
        return self.fields[self.names[0]] if len(self.names) == 1 else dict(self.fields)

    @state.setter
    def state(self, value: Any) -> None:
        if isinstance(value, dict):
            self.fields = {name: _as_field(value[name], self.problem.domain, self.fields[name]) for name in self.names}
        else:
            self.fields = {self.names[0]: _as_field(value, self.problem.domain, self.fields[self.names[0]])}
        self._history = self.scheme.new_history()

    # -- operators -----------------------------------------------------------

    def _signature(self, name: str) -> tuple:
        op = self.problem.linear_operators[name]
        if all(_is_number(v) for v in op.values()):
            return ("constant", *tuple(sorted((k, complex(v)) for k, v in op.items())))
        return ("variable", name)

    def implicit_operator(self, name: str, gamma: float) -> HPSOperator:
        """Factored operator ``I - gamma dt L_name`` (built on first use)."""
        key = (self._signature(name), round(gamma * self.dt, 14))
        op = self._operators.get(key)
        if op is None:
            coeffs = _implicit_operator(self.problem.linear_operators[name], gamma * self.dt)
            op = _make_operator(self.problem.domain, coeffs, 0.0, rankdef=self.problem.rankdef, **self._operator_kwargs)
            self._operators[key] = op
        return op

    @property
    def operator(self) -> HPSOperator:
        """Implicit operator of the first variable for the scheme's steady-state coefficient."""
        return self.implicit_operator(self.names[0], self.scheme.gamma)

    def build(self) -> SurfaceIVPSolver:
        """Factor the steady-state implicit operators now (otherwise done on first use)."""
        for name in self.names:
            self.implicit_operator(name, self.scheme.gamma).build()
        return self

    # -- step context used by the schemes --------------------------------------

    def solve(self, gamma: float, rhs: dict[str, Any]) -> dict[str, Any]:
        """Solve ``(I - gamma dt L) u = rhs`` for every variable."""
        groups: dict[int, tuple[HPSOperator, list[str]]] = {}
        for name in self.names:
            op = self.implicit_operator(name, gamma)
            groups.setdefault(id(op), (op, []))[1].append(name)
        out: dict[str, Any] = {}
        for op, names in groups.values():
            if len(names) == 1:
                out[names[0]] = op.solve(rhs[names[0]])
            else:
                for name, value in zip(names, op.solve_many([rhs[n] for n in names])):
                    out[name] = value
        return out

    def explicit(self, state: dict[str, Any], t: float) -> dict[str, Any]:
        return self.problem.explicit(state, t)

    def apply_linear(self, state: dict[str, Any]) -> dict[str, Any]:
        return {name: unified.apply_operator(self.problem.linear_operators[name], state[name]) for name in self.names}

    # -- stepping ------------------------------------------------------------

    def rhs(self) -> Any:
        """Right-hand side of an IMEX Euler step from the current state."""
        return self.problem.step_rhs(self.state, self.t, self.dt)

    def step(self, dt: float | None = None, rhs: Any = None) -> Any:
        """Advance one step and return the new state.

        ``rhs`` overrides the right-hand side of an IMEX Euler step (schemes
        ``sbdf1``/``rk111`` only), e.g. to couple fields by hand.
        """
        if dt is not None:
            if dt <= 0:
                raise ValueError("dt must be positive")
            if float(dt) != self.dt:
                self.dt = float(dt)
                self._history = self.scheme.new_history()
        if rhs is not None:
            if self.scheme.order != 1 or self.scheme.gamma != 1.0:
                raise ValueError("an explicit rhs override is only supported by the first-order schemes sbdf1/rk111")
            rhs_dict = rhs if isinstance(rhs, dict) else {self.names[0]: rhs}
            new = self.solve(1.0, rhs_dict)
        else:
            new = self.scheme.step(self, self.fields, self.t, self.dt, self._history)
        self.fields = {name: new[name] for name in self.names}
        self.t += self.dt
        self.iteration += 1
        if isinstance(self.scheme, SBDF) and self.scheme.order > 1 and len(self._history.states) == self.scheme.order:
            self._release_startup_operators()
        return self.state

    def _release_startup_operators(self) -> None:
        keep = round(self.scheme.gamma * self.dt, 14)
        for key in [k for k in self._operators if k[1] != keep]:
            del self._operators[key]

    def evaluator_state(self) -> dict[str, Any]:
        """State dictionary for evaluator callbacks."""
        out: dict[str, Any] = dict(self.fields)
        out.update({"state": self.state, "t": self.t, "iteration": self.iteration, "solver": self})
        if len(self.names) == 1:
            out["u"] = self.state
        return out

    def run(self, steps: int, evaluator: Any | None = None, evaluate_initial: bool = False) -> Any:
        """Advance ``steps`` steps, optionally evaluating outputs after each one."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        if evaluator is not None and evaluate_initial:
            evaluator.evaluate(self.iteration, time=self.t, state=self.evaluator_state())
        for _ in range(int(steps)):
            self.step()
            if evaluator is not None:
                evaluator.evaluate(self.iteration, time=self.t, state=self.evaluator_state())
        return self.state
