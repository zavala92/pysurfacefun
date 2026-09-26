"""
Contiguous storage and arithmetic shared by quadrilateral and triangular fields.

A scalar surface field stores the nodal values of all ``P`` patches in one
array ``data`` of shape ``(P,) + patch_shape``.  Arithmetic, NumPy ufuncs, and
reductions therefore run as single vectorized NumPy calls, independent of the
number of patches.  For compatibility with the original list-based API,
``field.vals`` is a list of per-patch *views* into ``data``; assigning
``field.vals[k] = values`` writes through to the underlying storage.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Number
from typing import Any

import numpy as np

Array = np.ndarray


class PatchValues(list):
    """List of per-patch arrays that are views into a field's contiguous data.

    Element assignment writes into the field; the number of patches is fixed.
    """

    __slots__ = ("_owner",)

    def __init__(self, owner: PatchField):
        super().__init__(owner._data)
        self._owner = owner

    def __setitem__(self, index, value):  # type: ignore[override]
        owner = self._owner
        if isinstance(index, slice):
            positions = range(*index.indices(len(self)))
            values = list(value)
            if len(values) != len(positions):
                raise ValueError("cannot change the number of patches of a surface field")
            for pos, item in zip(positions, values):
                self[pos] = item
            return
        value = np.asarray(value)
        owner._ensure_dtype(value.dtype)
        owner._data[index] = value.reshape(owner._data.shape[1:]) if value.size == owner._data[index].size else value
        list.__setitem__(self, index, owner._data[index])

    def _fixed_length(self, *args, **kwargs):
        raise TypeError("the number of patches of a surface field is fixed")

    append = extend = insert = pop = remove = clear = _fixed_length  # type: ignore[assignment]
    __delitem__ = __iadd__ = __imul__ = _fixed_length  # type: ignore[assignment]


def as_patch_data(vals: Any, npatches: int, patch_shape: tuple[int, ...]) -> Array:
    """Convert per-patch values into a ``(npatches,) + patch_shape`` array.

    ``vals`` may be an array of that shape (stored without copying), a list of
    per-patch arrays (each broadcast to ``patch_shape``, so scalars are fine),
    or a flat array with one row per patch.
    """
    shape = (npatches, *tuple(patch_shape))
    if isinstance(vals, np.ndarray):
        if vals.shape == shape:
            return vals
        if vals.size == int(np.prod(shape)) and vals.ndim >= 1 and vals.shape[0] == npatches:
            return vals.reshape(shape)
        raise ValueError(f"expected field values of shape {shape}, got {vals.shape}")
    items = list(vals)
    if len(items) != npatches:
        raise ValueError(f"expected values for {npatches} patches, got {len(items)}")
    if not items:
        return np.zeros(shape)
    arrays = [np.asarray(v) for v in items]
    dtype = np.result_type(*arrays)
    if not (np.issubdtype(dtype, np.number) or dtype == np.bool_):
        raise TypeError("surface field values must be numeric")
    out = np.empty(shape, dtype=dtype)
    size = int(np.prod(patch_shape))
    for k, arr in enumerate(arrays):
        out[k] = arr.reshape(patch_shape) if arr.size == size else np.broadcast_to(arr, patch_shape)
    return out


class PatchField:
    """Scalar field sampled on every node of a patch-based surface mesh.

    Subclasses define :meth:`patch_shape`; the domain provides ``npatches``,
    per-patch coordinates ``x, y, z``, and ``quadrature_weights``.
    """

    __array_priority__ = 1000
    _family = "patch"

    def __init__(self, domain: Any, vals: Any):
        self.domain = domain
        self._data = as_patch_data(vals, domain.npatches, self.patch_shape(domain))
        self._vals_cache: PatchValues | None = None

    # -- storage -------------------------------------------------------------

    @staticmethod
    def patch_shape(domain: Any) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def data(self) -> Array:
        """Nodal values of all patches, shape ``(P,) + patch_shape``."""
        return self._data

    @data.setter
    def data(self, values: Any) -> None:
        self._data = as_patch_data(np.asarray(values), self.domain.npatches, self.patch_shape(self.domain))
        self._vals_cache = None

    @property
    def vals(self) -> PatchValues:
        """Per-patch values as a list of views into :attr:`data`."""
        if self._vals_cache is None:
            self._vals_cache = PatchValues(self)
        return self._vals_cache

    @vals.setter
    def vals(self, values: Any) -> None:
        self.data = as_patch_data(values, self.domain.npatches, self.patch_shape(self.domain))

    def _ensure_dtype(self, dtype: np.dtype) -> None:
        new = np.result_type(self._data.dtype, dtype)
        if new != self._data.dtype:
            self._data = self._data.astype(new)
            if self._vals_cache is not None:
                list.__init__(self._vals_cache, self._data)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def npatches(self) -> int:
        return int(self._data.shape[0])

    def __array__(self, dtype=None, copy=None) -> Array:
        if copy:
            return np.array(self._data, dtype=dtype, copy=True)
        return np.asarray(self._data, dtype=dtype)

    def __repr__(self) -> str:
        shape = "x".join(str(s) for s in self._data.shape[1:])
        return f"{type(self).__name__}(npatches={self.npatches}, patch_shape={shape}, dtype={self.dtype})"

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_callable(cls, domain: Any, func: Callable[[Array, Array, Array], Any]):
        """Sample ``func(x, y, z)`` patch by patch."""
        return cls(domain, [np.asarray(func(x, y, z)) for x, y, z in zip(domain.x, domain.y, domain.z)])

    @classmethod
    def constant(cls, domain: Any, value: Number):
        """Field equal to ``value`` everywhere."""
        shape = (domain.npatches, *cls.patch_shape(domain))
        return cls(domain, np.full(shape, value, dtype=np.result_type(value, float)))

    def copy(self):
        """Deep copy of the nodal values (the domain is shared)."""
        return type(self)(self.domain, self._data.copy())

    def _new(self, data: Array):
        return type(self)(self.domain, data)

    # -- reductions ----------------------------------------------------------

    def integral(self, reduce: bool = True):
        """Surface integral (per-patch integrals when ``reduce=False``)."""
        weights = self.domain.quadrature_weights
        per_patch = (self._data * weights).reshape(self.npatches, -1).sum(axis=1)
        if not reduce:
            return per_patch
        total = per_patch.sum()
        return float(np.real(total)) if np.isrealobj(total) else complex(total)

    def norm_inf(self) -> float:
        """Maximum absolute nodal value."""
        return float(np.max(np.abs(self._data))) if self._data.size else 0.0

    def mean2(self):
        """Surface average."""
        return self.integral() / float(np.sum(self.domain.quadrature_weights))

    def remove_mean(self):
        """Field minus its surface average."""
        return self - self.mean2()

    # -- arithmetic ----------------------------------------------------------

    def _operand(self, other: Any) -> Any:
        if isinstance(other, PatchField):
            if other._family != self._family:
                raise TypeError(f"cannot combine a {type(self).__name__} with a {type(other).__name__}")
            if other._data.shape[:1] != self._data.shape[:1] or other._data.shape[1:] != self._data.shape[1:]:
                raise ValueError("surface fields live on incompatible meshes")
            return other._data
        if isinstance(other, PatchVectorField):
            return NotImplemented
        if isinstance(other, (Number, np.ndarray, np.generic)):
            return other
        return NotImplemented

    def _binary(self, other: Any, op: Callable, reflected: bool = False):
        value = self._operand(other)
        if value is NotImplemented:
            return NotImplemented
        return self._new(op(value, self._data) if reflected else op(self._data, value))

    def __add__(self, other):
        return self._binary(other, np.add)

    def __radd__(self, other):
        return self._binary(other, np.add, True)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        return self._binary(other, np.subtract, True)

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    def __rmul__(self, other):
        return self._binary(other, np.multiply, True)

    def __truediv__(self, other):
        return self._binary(other, np.true_divide)

    def __rtruediv__(self, other):
        return self._binary(other, np.true_divide, True)

    def __pow__(self, other):
        return self._binary(other, np.power)

    def __rpow__(self, other):
        return self._binary(other, np.power, True)

    def __neg__(self):
        return self._new(-self._data)

    def __pos__(self):
        return self._new(+self._data)

    def __abs__(self):
        return self._new(np.abs(self._data))

    @property
    def real(self):
        """Real part."""
        return self._new(np.real(self._data))

    @property
    def imag(self):
        """Imaginary part."""
        return self._new(np.imag(self._data))

    def conj(self):
        """Complex conjugate."""
        return self._new(np.conj(self._data))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.get("out") is not None:
            return NotImplemented
        if method != "__call__":
            # Reductions and friends act on the nodal data and return plain arrays/scalars.
            data = [item.data if isinstance(item, PatchField) else item for item in inputs]
            return getattr(ufunc, method)(*data, **kwargs)
        args = []
        for item in inputs:
            if isinstance(item, PatchField):
                value = self._operand(item)
                if value is NotImplemented:
                    return NotImplemented
                args.append(value)
            elif isinstance(item, PatchVectorField):
                return NotImplemented
            else:
                args.append(item)
        result = ufunc(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(self._new(r) for r in result)
        return self._new(result)


class PatchVectorField:
    """Three-component vector field; components are scalar :class:`PatchField` objects."""

    __array_ufunc__ = None
    _scalar_family = "patch"

    def __init__(self, components):
        components = tuple(components)
        if len(components) != 3:
            raise ValueError("vector fields have exactly three components")
        self.components = components

    @property
    def domain(self):
        return self.components[0].domain

    def __iter__(self):
        return iter(self.components)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.components[0]!r} x 3)"

    def copy(self):
        return type(self)(tuple(c.copy() for c in self.components))

    def norm_inf(self) -> float:
        """Maximum pointwise Euclidean length."""
        a, b, c = (comp.data for comp in self.components)
        return float(np.max(np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2 + np.abs(c) ** 2)))

    def _vector_operand(self, other: Any):
        if isinstance(other, PatchVectorField):
            return other.components
        arr = np.asarray(other) if not isinstance(other, PatchField) else None
        if arr is not None and arr.ndim == 1 and arr.size == 3:
            return tuple(arr[i] for i in range(3))
        return None

    def __add__(self, other):
        parts = self._vector_operand(other)
        if parts is not None:
            return type(self)(tuple(a + b for a, b in zip(self.components, parts)))
        if isinstance(other, (Number, np.generic)) or (isinstance(other, np.ndarray) and other.ndim == 0):
            return type(self)(tuple(a + other for a in self.components))
        raise ValueError("can only add scalars, length-3 vectors, or vector fields")

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other if not isinstance(other, (list, tuple)) else -np.asarray(other))

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return type(self)(tuple(-a for a in self.components))

    def __mul__(self, other):
        return type(self)(tuple(a * other for a in self.components))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return type(self)(tuple(a / other for a in self.components))
