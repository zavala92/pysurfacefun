"""Benchmark the fast direct solver: leaf assembly, factorization, and repeated solves.

Usage::

    python benchmarks/benchmark_solver.py            # full set of meshes
    python benchmarks/benchmark_solver.py --quick    # small smoke test
    python benchmarks/benchmark_solver.py --compare-orderings

For each mesh the script reports the time to assemble the leaf operators, the
time to factor (merge) them, the time per repeated solve (the cost of one
implicit time step), and statistics of the merge tree.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pysurfacefun as psf


def cases(quick: bool):
    if quick:
        yield "quad sphere n=8 nref=1", lambda: psf.sphere(8, 1)
        yield "tri icosphere n=7 nref=1", lambda: psf.icosphere_tri(7, 1)
        return
    yield "quad sphere n=12 nref=2", lambda: psf.sphere(12, 2)
    yield "quad sphere n=17 nref=3", lambda: psf.sphere(17, 3)
    yield "quad torus n=9 16x32", lambda: psf.torus(9, 16, 32)
    cow = ROOT / "notebook_data" / "cow.csv"
    if cow.exists():
        yield "quad cow (Rhino) n=8", lambda: psf.from_rhino(str(cow), 8)
        yield "quad cow (Rhino) n=12", lambda: psf.resample_mesh(psf.from_rhino(str(cow), 8), 12)
    yield "tri icosphere n=9 nref=3", lambda: psf.icosphere_tri(9, 3)
    yield "tri icosphere n=17 nref=2", lambda: psf.icosphere_tri(17, 2)


def benchmark(label: str, dom, strategy: str, repeats: int) -> dict:
    op = {"lap": -1e-2, "c": 1.0}
    rhs = psf.field(lambda x, y, z: np.cos(3 * x) * y + z, dom)
    t0 = time.perf_counter()
    L = psf.surfaceop(dom, op, merge_strategy=strategy)
    t1 = time.perf_counter()
    L.build()
    t2 = time.perf_counter()
    L.solve(rhs)
    t3 = time.perf_counter()
    for _ in range(repeats):
        L.solve(rhs)
    t4 = time.perf_counter()
    stats = L.stats()
    return {
        "label": label,
        "strategy": strategy,
        "patches": dom.npatches,
        "n": dom.n,
        "leaves": t1 - t0,
        "factor": t2 - t1,
        "first_solve": t3 - t2,
        "solve": (t4 - t3) / repeats,
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small meshes only")
    parser.add_argument("--repeats", type=int, default=10, help="repeated solves to time")
    parser.add_argument("--compare-orderings", action="store_true", help="also time the legacy natural merge order")
    args = parser.parse_args()

    strategies = ["nested_dissection", "natural"] if args.compare_orderings else ["nested_dissection"]
    header = (
        f"{'mesh':28s} {'order':>17s} {'P':>5s} {'n':>3s} {'leaves[s]':>9s} {'factor[s]':>9s} "
        f"{'solve[ms]':>9s} {'max sep':>7s} {'GFlop':>7s}"
    )
    print(f"pysurfacefun {psf.__version__}, NumPy {np.__version__}")
    print(header)
    print("-" * len(header))
    for label, make in cases(args.quick):
        dom = make()
        for strategy in strategies:
            r = benchmark(label, dom, strategy, args.repeats)
            print(
                f"{r['label']:28s} {r['strategy']:>17s} {r['patches']:5d} {r['n']:3d} {r['leaves']:9.2f} "
                f"{r['factor']:9.2f} {1e3 * r['solve']:9.2f} {r['max_separator']:7d} {r['factor_gflop']:7.2f}"
            )


if __name__ == "__main__":
    main()
