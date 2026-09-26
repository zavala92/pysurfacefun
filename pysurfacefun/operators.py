"""
Partial differential operators and the shared driver for patch-based solvers.

A scalar second-order surface operator is written in terms of the tangential
Cartesian derivatives ``D_x, D_y, D_z`` as

.. math::

    L u = \\sum_{c,d} a_{cd} D_c D_d u + \\sum_c b_c D_c u + b_0 u,

where each coefficient may be a constant, a callable ``f(x, y, z)``, or a
surface field.  :class:`PDO` stores the thirteen coefficients and
:func:`parse_pdo` builds one from a dictionary such as
``{"lap": 1.0, "c": 3.0}``.

:class:`HPSOperator` implements the user-facing solve/update API on top of
:class:`pysurfacefun.hps.HPSSolver`; the quadrilateral and triangular
operators only provide the dense leaf operators and the conversions between
surface fields and nodal arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Number
from typing import Any

import numpy as np

from .hps import (
    HPSSolver,
    LeafOperators,
    MergeTree,
    Patch,
    merge_idx_from_tree,
    tree_from_merge_idx,
    validate_tree,
)

Array = np.ndarray

#: Second-order PDO coefficient names and their ``(outer, inner)`` derivative axes.
SECOND_ORDER_TERMS: dict[str, tuple[int, int]] = {
    "dxx": (0, 0),
    "dxy": (0, 1),
    "dxz": (0, 2),
    "dyx": (1, 0),
    "dyy": (1, 1),
    "dyz": (1, 2),
    "dzx": (2, 0),
    "dzy": (2, 1),
    "dzz": (2, 2),
}
#: First-order PDO coefficient names and their derivative axis.
FIRST_ORDER_TERMS: dict[str, int] = {"dx": 0, "dy": 1, "dz": 2}


@dataclass
class PDO:
    """Coefficients of a scalar second-order surface operator.

    ``dxy`` multiplies ``D_x D_y u`` (``D_y`` applied first).  ``b`` is the
    zeroth-order coefficient.  Each coefficient may be a number, a callable
    ``f(x, y, z)``, or a surface field on the operator's domain.
    """

    dxx: Any = 0.0
    dyy: Any = 0.0
    dzz: Any = 0.0
    dxy: Any = 0.0
    dyx: Any = 0.0
    dyz: Any = 0.0
    dzy: Any = 0.0
    dxz: Any = 0.0
    dzx: Any = 0.0
    dx: Any = 0.0
    dy: Any = 0.0
    dz: Any = 0.0
    b: Any = 0.0

    def items(self):
        """Iterate over ``(name, coefficient)`` pairs."""
        for item in fields(self):
            yield item.name, getattr(self, item.name)

    def nonzero(self) -> dict[str, Any]:
        """Coefficients that are not identically zero."""
        return {name: value for name, value in self.items() if not is_zero(value)}


def is_zero(value: Any) -> bool:
    """True for ``None`` and scalar zeros (fields and callables are never zero)."""
    if value is None:
        return True
    if isinstance(value, Number) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return bool(value == 0)
    return False


def _is_field(value: Any) -> bool:
    return hasattr(value, "domain") and hasattr(value, "data")


class CoefficientSum:
    """Sum of coefficients of mixed kinds (numbers, callables, fields), evaluated on demand."""

    def __init__(self, parts: list[Any]):
        self.parts = parts

    def __repr__(self) -> str:
        return f"CoefficientSum({self.parts!r})"


def _add(left: Any, right: Any) -> Any:
    if is_zero(left):
        return right
    if is_zero(right):
        return left
    mixed = any((callable(v) and not _is_field(v)) or isinstance(v, CoefficientSum) for v in (left, right))
    if mixed:
        parts: list[Any] = []
        for v in (left, right):
            parts.extend(v.parts if isinstance(v, CoefficientSum) else [v])
        return CoefficientSum(parts)
    return left + right


def parse_pdo(op: dict[str, Any] | PDO) -> PDO:
    """Build a :class:`PDO` from a dictionary of coefficients.

    Recognized keys are the :class:`PDO` field names plus the shorthands
    ``"lap"`` (adds to ``dxx``, ``dyy``, ``dzz``), ``"grad"`` (adds to ``dx``,
    ``dy``, ``dz``), and ``"c"`` (adds to ``b``).  Repeated contributions are
    summed, so ``{"lap": 1, "dxx": 2}`` gives ``dxx = 3``.
    """
    if isinstance(op, PDO):
        return PDO(**dict(op.items()))
    out = PDO()
    names = {item.name for item in fields(PDO)}
    for key, value in op.items():
        if key == "lap":
            targets = ("dxx", "dyy", "dzz")
        elif key == "grad":
            targets = ("dx", "dy", "dz")
        elif key == "c":
            targets = ("b",)
        elif key in names:
            targets = (key,)
        else:
            raise ValueError(f"unknown operator term {key!r}")
        for name in targets:
            setattr(out, name, _add(getattr(out, name), value))
    return out


# ---------------------------------------------------------------------------
# Shared solver driver.
# ---------------------------------------------------------------------------


class HPSOperator:
    """Fast direct solver for ``L u = f`` on a patch surface mesh.

    Subclasses provide the dense leaf operators and the conversions between
    surface fields and nodal arrays.  Construction assembles the leaves; the
    first :meth:`solve` (or an explicit :meth:`build`) merges them.  The
    factorization does not depend on the right-hand side, so :meth:`solve`
    with new data only runs the fast upward/downward sweeps.

    Parameters
    ----------
    domain:
        Surface mesh.
    op:
        Operator coefficients, e.g. ``{"lap": 1.0, "c": 2.0}``.
    rhs:
        Right-hand side: a field on ``domain``, a callable ``f(x, y, z)``, or
        a number.
    merge_idx:
        Optional legacy level-by-level merge schedule.
    merge_strategy:
        ``"nested_dissection"`` (default), ``"natural"``, or ``"matching"``
        (``"default"`` and ``"adjacent"`` are accepted aliases).
    rankdef:
        Regularize the constant null space of a pure Laplace--Beltrami
        operator on a closed surface.
    """

    def __init__(
        self,
        domain: Any,
        op: dict[str, Any] | PDO,
        rhs: Any = 0.0,
        merge_idx: list[list[tuple[int, int | None]]] | None = None,
        merge_strategy: str = "nested_dissection",
        *,
        merge_tree: MergeTree | None = None,
        rankdef: bool = False,
    ):
        self.domain = domain
        self.op = parse_pdo(op)
        self.rhs = rhs
        self.rankdef = rankdef
        self.merge_strategy = merge_strategy.lower()
        self._tree: MergeTree | None = merge_tree
        if merge_idx is not None:
            self._tree = tree_from_merge_idx(merge_idx, domain.npatches)
        self.solver = HPSSolver(self._build_leaf_operators(), tree=None, lazy_tree=True)
        if self._tree is not None:
            validate_tree(self._tree, domain.npatches)
        self._built = False

    # -- hooks ---------------------------------------------------------------

    def _build_leaf_operators(self) -> LeafOperators:
        raise NotImplementedError

    def _rhs_values(self, rhs: Any) -> Array:
        """Right-hand side at the interior nodes, shape ``(P, m)``."""
        raise NotImplementedError

    def _field_from_nodal(self, U: Array) -> Any:
        """Surface field from nodal values of shape ``(P, N)``."""
        raise NotImplementedError

    # -- merge tree ----------------------------------------------------------

    @property
    def merge_tree(self) -> MergeTree:
        """Binary merge tree used by the factorization."""
        if self._tree is None:
            self._tree = self.solver.default_tree(self.merge_strategy)
        return self._tree

    @merge_tree.setter
    def merge_tree(self, tree: MergeTree) -> None:
        if self._built:
            raise RuntimeError("the merge tree cannot change after build()")
        validate_tree(tree, self.domain.npatches)
        self._tree = tree

    @property
    def merge_idx(self) -> list[list[tuple[int, int | None]]]:
        """Merge tree in the legacy level-by-level format."""
        return merge_idx_from_tree(self.merge_tree, self.domain.npatches)

    @merge_idx.setter
    def merge_idx(self, merge_idx: list[list[tuple[int, int | None]]]) -> None:
        self.merge_tree = tree_from_merge_idx(merge_idx, self.domain.npatches)

    @property
    def patches(self) -> list[Patch]:
        """Leaf patches before :meth:`build`, the root afterwards."""
        if self.solver.root is not None:
            return [self.solver.root]
        return list(self.solver.leaves)

    # -- solving -------------------------------------------------------------

    @property
    def built(self) -> bool:
        return self._built

    def build(self) -> HPSOperator:
        """Merge the leaf operators (idempotent)."""
        if not self._built:
            self.solver.tree = self.merge_tree
            self.solver.factor(rankdef=self.rankdef)
            self._built = True
        return self

    def update_rhs(self, rhs: Any) -> HPSOperator:
        """Replace the right-hand side; the factorization is reused."""
        self.rhs = rhs
        return self

    def solve(self, rhs: Any = None, bc: Any = None) -> Any:
        """Solve ``L u = f``.

        Parameters
        ----------
        rhs:
            New right-hand side (optional; defaults to the stored one).
        bc:
            Dirichlet data on the boundary of an open surface: a callable
            ``g(x, y, z)`` or an array of values at the boundary skeleton
            points ``self.patches[0].xyz``.  Closed surfaces ignore it.
        """
        if rhs is not None:
            self.rhs = rhs
        self.build()
        F = self._rhs_values(self.rhs)
        bc_values = None
        if bc is not None:
            root = self.solver.root
            assert root is not None
            if callable(bc):
                bc_values = np.asarray(bc(root.xyz[:, 0], root.xyz[:, 1], root.xyz[:, 2]))
            else:
                bc_values = np.asarray(bc)
            bc_values = np.broadcast_to(bc_values, (root.nboundary,)).copy()
        return self._field_from_nodal(self.solver.solve(F, bc=bc_values))

    def solve_many(self, rhs_list: list[Any]) -> list[Any]:
        """Solve for several right-hand sides with one sweep of the factored tree."""
        self.build()
        F = np.stack([np.asarray(self._rhs_values(rhs)) for rhs in rhs_list], axis=-1)
        U = self.solver.solve(F)
        return [self._field_from_nodal(U[:, :, k]) for k in range(len(rhs_list))]

    def apply(self, rhs: Any) -> Any:
        """Update the right-hand side and immediately solve."""
        return self.solve(rhs)

    def stats(self) -> dict[str, float]:
        """Factorization statistics (separator sizes, flop and memory estimates)."""
        self.build()
        return self.solver.stats()


def coefficient_arrays(value: Any, domain: Any, shape: tuple[int, ...]) -> Array:
    """Evaluate a PDO coefficient on every node, returning an array of ``shape``.

    ``shape`` is ``(P,) + patch_shape``.  Accepts numbers, callables
    ``f(x, y, z)`` (evaluated patch by patch), fields on ``domain``, sums of
    these, and arrays of shape ``shape``, ``patch_shape`` (same values on every
    patch), or ``patch_shape + (P,)`` (legacy column layout).
    """
    if isinstance(value, CoefficientSum):
        return sum(coefficient_arrays(part, domain, shape) for part in value.parts)
    if _is_field(value):
        return np.asarray(value.data).reshape(shape)
    if callable(value):
        return np.stack(
            [np.broadcast_to(np.asarray(value(x, y, z)), shape[1:]) for x, y, z in zip(domain.x, domain.y, domain.z)]
        )
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.full(shape, arr.item(), dtype=np.result_type(arr, float))
    if arr.shape == shape:
        return arr
    if arr.shape == shape[1:]:
        return np.broadcast_to(arr, shape).copy()
    if arr.shape == (*shape[1:], shape[0]):
        return np.moveaxis(arr, -1, 0)
    if arr.size == int(np.prod(shape[1:])):
        return np.broadcast_to(arr.reshape(shape[1:]), shape).copy()
    raise ValueError("coefficient arrays must provide nodal values on the mesh")


def evaluate_rhs_nodes(rhs: Any, domain: Any, flat_index: Array, nodes_per_patch: int) -> Array:
    """Right-hand side values at selected nodes of every patch, shape ``(P, len(flat_index))``.

    ``flat_index`` indexes each patch's nodal values flattened in C order.
    """
    npatches = domain.npatches
    if _is_field(rhs):
        return np.asarray(rhs.data).reshape(npatches, nodes_per_patch)[:, flat_index]
    if callable(rhs):
        rows = []
        for x, y, z in zip(domain.x, domain.y, domain.z):
            xs = np.asarray(x).reshape(-1)[flat_index]
            ys = np.asarray(y).reshape(-1)[flat_index]
            zs = np.asarray(z).reshape(-1)[flat_index]
            rows.append(np.broadcast_to(np.asarray(rhs(xs, ys, zs)).reshape(-1), xs.shape))
        return np.stack(rows)
    arr = np.asarray(rhs)
    if arr.ndim == 0:
        return np.full((npatches, flat_index.size), arr.item(), dtype=np.result_type(arr, float))
    raise TypeError("rhs must be a surface field, a callable f(x, y, z), or a number")
