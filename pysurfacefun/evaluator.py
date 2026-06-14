"""Evaluation tasks and repeatable output handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
from numbers import Number
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import numpy as np

from .core import SurfaceFunction, SurfaceVectorFunction, write_vtu
from .tri import TriangleSurfaceFunction, TriangleSurfaceVectorFunction, write_tri_vtu


State = Mapping[str, Any]
TaskCallback = Callable[[State], Any] | Callable[[], Any]
SurfaceValue = SurfaceFunction | TriangleSurfaceFunction
SurfaceVectorValue = SurfaceVectorFunction | TriangleSurfaceVectorFunction


def _safe_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return out.strip("._") or "value"


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    return None


def _call_value(value: Any, state: State) -> Any:
    if not callable(value):
        return value

    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return value(state)

    required = [
        param
        for param in signature.parameters.values()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if required:
        return value(state)
    return value()


def _patch_arrays(prefix: str, vals: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}_patch_{k:04d}": np.asarray(val) for k, val in enumerate(vals)}


@dataclass
class EvaluationTask:
    """Named value to evaluate at a fixed iteration cadence."""

    name: str
    value: TaskCallback | Any
    every: int = 1

    def __post_init__(self) -> None:
        if self.every <= 0:
            raise ValueError("every must be positive")

    def ready(self, iteration: int, force: bool = False) -> bool:
        return force or iteration % self.every == 0

    def evaluate(self, state: State) -> Any:
        return _call_value(self.value, state)


@dataclass
class EvaluationRecord:
    """Values and files produced for one evaluator call."""

    iteration: int
    time: float | None
    values: dict[str, Any]
    files: list[dict[str, Any]] = field(default_factory=list)


class OutputHandler:
    """Base class for evaluator output handlers."""

    def write(self, evaluator: "Evaluator", record: EvaluationRecord) -> list[dict[str, Any]]:
        raise NotImplementedError


class JSONLinesOutputHandler(OutputHandler):
    """Write scalar diagnostics to a JSON lines file."""

    def __init__(self, filename: str = "scalars.jsonl", append: bool = False):
        self.filename = filename
        self.append = append
        self._written_paths: set[Path] = set()

    def write(self, evaluator: "Evaluator", record: EvaluationRecord) -> list[dict[str, Any]]:
        values = {}
        for name, value in record.values.items():
            json_value = _json_ready(value)
            if json_value is not None:
                values[name] = json_value

        if not values:
            return []

        path = evaluator.output_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"iteration": record.iteration, "time": record.time, "values": values}
        mode = "a" if self.append or path in self._written_paths else "w"
        with path.open(mode, encoding="utf-8") as fid:
            fid.write(json.dumps(payload, sort_keys=True) + "\n")
        self._written_paths.add(path)
        return [{"kind": "jsonl", "path": evaluator.relative_path(path), "values": sorted(values)}]


class NPZOutputHandler(OutputHandler):
    """Write evaluated arrays and fields to compressed ``.npz`` snapshots."""

    def __init__(
        self,
        directory: str = "arrays",
        filename_template: str = "{prefix}_{name}_{iteration:06d}.npz",
        include_geometry: bool = True,
        compressed: bool = True,
    ):
        self.directory = directory
        self.filename_template = filename_template
        self.include_geometry = include_geometry
        self.compressed = compressed

    def write(self, evaluator: "Evaluator", record: EvaluationRecord) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        for name, value in record.values.items():
            arrays = self._arrays_for(value)
            if not arrays:
                continue
            filename = self.filename_template.format(
                prefix=evaluator.prefix,
                name=_safe_name(name),
                iteration=record.iteration,
            )
            path = evaluator.output_dir / self.directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = np.savez_compressed if self.compressed else np.savez
            writer(path, **arrays)
            files.append({"kind": "npz", "path": evaluator.relative_path(path), "task": name})
        return files

    def _arrays_for(self, value: Any) -> dict[str, np.ndarray]:
        if isinstance(value, (SurfaceFunction, TriangleSurfaceFunction)):
            arrays = {
                "kind": np.asarray(type(value).__name__),
                "npatches": np.asarray(value.domain.npatches),
                **_patch_arrays("value", value.vals),
            }
            if self.include_geometry:
                arrays.update(_patch_arrays("x", value.domain.x))
                arrays.update(_patch_arrays("y", value.domain.y))
                arrays.update(_patch_arrays("z", value.domain.z))
            return arrays

        if isinstance(value, (SurfaceVectorFunction, TriangleSurfaceVectorFunction)):
            arrays = {
                "kind": np.asarray(type(value).__name__),
                "npatches": np.asarray(value.domain.npatches),
            }
            for component, field_value in zip(("x", "y", "z"), value.components):
                arrays.update(_patch_arrays(f"value_{component}", field_value.vals))
            if self.include_geometry:
                arrays.update(_patch_arrays("x", value.domain.x))
                arrays.update(_patch_arrays("y", value.domain.y))
                arrays.update(_patch_arrays("z", value.domain.z))
            return arrays

        if isinstance(value, (Number, np.ndarray, np.generic)):
            return {"value": np.asarray(value)}

        return {}


class VTKOutputHandler(OutputHandler):
    """Write scalar or vector field snapshots as VTU files."""

    def __init__(
        self,
        directory: str = "vtk",
        filename_template: str = "{prefix}_{name}_{iteration:06d}.vtu",
        point_name: str | None = None,
        nvis: int | None = None,
        skip_missing_meshio: bool = False,
    ):
        self.directory = directory
        self.filename_template = filename_template
        self.point_name = point_name
        self.nvis = nvis
        self.skip_missing_meshio = skip_missing_meshio

    def write(self, evaluator: "Evaluator", record: EvaluationRecord) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []
        for name, value in record.values.items():
            if isinstance(value, (SurfaceFunction, TriangleSurfaceFunction)):
                files.extend(self._write_scalar(evaluator, record.iteration, name, value))
            elif isinstance(value, (SurfaceVectorFunction, TriangleSurfaceVectorFunction)):
                for suffix, component in zip(("x", "y", "z"), value.components):
                    files.extend(self._write_scalar(evaluator, record.iteration, f"{name}_{suffix}", component))
        return files

    def _write_scalar(
        self,
        evaluator: "Evaluator",
        iteration: int,
        name: str,
        value: SurfaceValue,
    ) -> list[dict[str, Any]]:
        filename = self.filename_template.format(
            prefix=evaluator.prefix,
            name=_safe_name(name),
            iteration=iteration,
        )
        path = evaluator.output_dir / self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        point_name = self.point_name or _safe_name(name)

        if isinstance(value, TriangleSurfaceFunction):
            write_tri_vtu(str(path), value, point_name=point_name, nvis=self.nvis)
        else:
            try:
                write_vtu(str(path), value, point_name=point_name)
            except ImportError:
                if not self.skip_missing_meshio:
                    raise
                return [
                    {
                        "kind": "vtu",
                        "path": evaluator.relative_path(path),
                        "task": name,
                        "status": "skipped",
                        "reason": "meshio is not installed",
                    }
                ]

        return [{"kind": "vtu", "path": evaluator.relative_path(path), "task": name}]


class Evaluator:
    """Evaluate named tasks and write repeatable output records."""

    def __init__(
        self,
        output_dir: str | Path = "notebook_outputs",
        prefix: str = "run",
        handlers: list[OutputHandler] | None = None,
        metadata: Mapping[str, Any] | None = None,
        manifest: str = "manifest.json",
    ):
        self.output_dir = Path(output_dir)
        self.prefix = _safe_name(prefix)
        self.handlers = list(handlers) if handlers is not None else [JSONLinesOutputHandler()]
        self.metadata = dict(metadata or {})
        self.manifest = manifest
        self.tasks: list[EvaluationTask] = []
        self.history: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_task(self, name: str, value: TaskCallback | Any, every: int = 1) -> EvaluationTask:
        """Register a task evaluated every ``every`` iterations."""
        task = EvaluationTask(name, value, every=every)
        self.tasks.append(task)
        return task

    def add_handler(self, handler: OutputHandler) -> OutputHandler:
        """Append an output handler and return it."""
        self.handlers.append(handler)
        return handler

    def evaluate(
        self,
        iteration: int = 0,
        time: float | None = None,
        state: State | None = None,
        force: bool = False,
    ) -> EvaluationRecord:
        """Evaluate ready tasks, run output handlers, and refresh the manifest."""
        state = {} if state is None else state
        iteration = int(iteration)
        values = {
            task.name: task.evaluate(state)
            for task in self.tasks
            if task.ready(iteration, force=force)
        }
        record = EvaluationRecord(iteration=iteration, time=time, values=values)
        for handler in self.handlers:
            record.files.extend(handler.write(self, record))
        self.history.append(self._history_entry(record))
        self.write_manifest()
        return record

    def relative_path(self, path: str | Path) -> str:
        """Return a path relative to ``output_dir`` when possible."""
        path = Path(path)
        try:
            return str(path.relative_to(self.output_dir))
        except ValueError:
            return str(path)

    def write_manifest(self) -> Path:
        """Write a JSON manifest describing tasks and generated files."""
        path = self.output_dir / self.manifest
        payload = {
            "prefix": self.prefix,
            "metadata": self.metadata,
            "tasks": [{"name": task.name, "every": task.every} for task in self.tasks],
            "history": self.history,
        }
        with path.open("w", encoding="utf-8") as fid:
            json.dump(payload, fid, indent=2, sort_keys=True)
            fid.write("\n")
        return path

    def _history_entry(self, record: EvaluationRecord) -> dict[str, Any]:
        values = {}
        for name, value in record.values.items():
            json_value = _json_ready(value)
            if json_value is not None:
                values[name] = json_value
        return {
            "iteration": record.iteration,
            "time": record.time,
            "values": values,
            "files": list(record.files),
        }
