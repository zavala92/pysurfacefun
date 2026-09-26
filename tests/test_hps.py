"""Fast direct solver: merge ordering, edge topology, factorization reuse, and boundary data."""

import numpy as np
import pytest

import pysurfacefun as psf
from pysurfacefun import hps
from pysurfacefun.core import chebpts2


def _relerr(a, b):
    return psf.norm(a - b, "inf") / psf.norm(b, "inf")


@pytest.mark.parametrize("make", [lambda: psf.sphere(7, 1), lambda: psf.icosphere_tri(6, 1)])
def test_all_merge_orders_give_the_same_solution(make):
    dom = make()
    f = psf.field(lambda x, y, z: x * y * z + 0.3 * x - 0.2, dom)
    op = {"lap": -1.0, "c": 3.0}
    reference = psf.surfaceop(dom, op, f, merge_strategy="natural").solve()
    for strategy in ("nested_dissection", "matching", "adjacent", "default"):
        u = psf.surfaceop(dom, op, f, merge_strategy=strategy).solve()
        assert _relerr(u, reference) < 1e-12, strategy


def test_custom_merge_schedule_returns_patches_in_mesh_order():
    # A schedule whose leaf order differs from 0..P-1 used to scramble the solution patches.
    dom = psf.icosphere_tri(6, 0)
    f = psf.field(lambda x, y, z: x + 2 * y * z, dom)
    reference = psf.surfaceop(dom, {"lap": -1.0, "c": 2.0}, f).solve()
    order = list(np.random.default_rng(3).permutation(dom.npatches))
    tree = order[0]
    for k in order[1:]:
        tree = (tree, k)  # deep chain: exercises the iterative traversals
    L = psf.surfaceop(dom, {"lap": -1.0, "c": 2.0}, f, merge_tree=tree)
    assert _relerr(L.solve(), reference) < 1e-12
    schedule = L.merge_idx
    assert hps.tree_leaves(hps.tree_from_merge_idx(schedule, dom.npatches)) == hps.tree_leaves(tree)


def test_merge_idx_setter_and_legacy_schedule():
    dom = psf.sphere(6, 1)
    L = psf.surfaceop(dom, {"lap": -1.0, "c": 1.0}, 1.0)
    L.merge_idx = psf.core.default_merge_idx(dom.npatches)
    u = L.solve()
    assert _relerr(u, psf.surfaceop(dom, {"lap": -1.0, "c": 1.0}, 1.0).solve()) < 1e-12
    with pytest.raises(RuntimeError):
        L.merge_idx = psf.core.default_merge_idx(dom.npatches)


@pytest.mark.parametrize("make", [lambda n: psf.torus(n, 2, 4), lambda n: psf.stellarator(n, 2, 4)])
def test_coarse_periodic_meshes_glue_the_right_edges(make):
    # With nu=2, two different curves share both end points; gluing by end points alone
    # produced errors of order 1e11 here.  The solution must converge spectrally.
    g = lambda x, y, z: np.sin(x) * np.cos(2 * y) + 0.3 * z * x
    errors = []
    for n in (12, 16, 20):
        dom = make(n)
        exact = psf.field(g, dom)
        rhs = psf.lap(exact)
        u = psf.surfaceop(dom, {"lap": 1.0}, rhs - psf.mean2(rhs), rankdef=True).solve().remove_mean()
        errors.append(_relerr(u, exact.remove_mean()))
    assert errors[-1] < errors[0] and errors[-1] < 5e-2


def test_nested_dissection_reduces_factorization_cost():
    dom = psf.from_rhino("notebook_data/cow.csv", 8)
    op = {"lap": -1e-3, "c": 1.0}
    natural = psf.surfaceop(dom, op, merge_strategy="natural").stats()
    nested = psf.surfaceop(dom, op).stats()
    assert nested["max_separator"] < natural["max_separator"] / 4
    assert nested["factor_gflop"] < natural["factor_gflop"] / 10
    assert nested["empty_merges"] < natural["empty_merges"]


def test_factorization_is_reused_across_right_hand_sides():
    dom = psf.sphere(8, 1)
    L = psf.surfaceop(dom, {"lap": -1.0, "c": 4.0})
    f1 = psf.field(lambda x, y, z: x, dom)
    f2 = psf.field(lambda x, y, z: y * z, dom)
    u1 = L.solve(f1)
    root = L.patches[0]
    u2 = L.solve(f2)
    assert L.patches[0] is root
    many = L.solve_many([f1, f2, f1 + 2j * f2])
    assert _relerr(many[0], u1) < 1e-13 and _relerr(many[1], u2) < 1e-13
    assert _relerr(many[2], u1 + 2j * u2) < 1e-13
    assert _relerr(L.apply(f1), u1) < 1e-13


def test_complex_rhs_at_construction():
    dom = psf.sphere(7, 1)
    f = psf.field(lambda x, y, z: (1 + 2j) * x + 1j * y * z, dom)
    L = psf.surfaceop(dom, {"lap": -0.3, "c": 1.0}, f)
    u = L.solve()
    re = psf.surfaceop(dom, {"lap": -0.3, "c": 1.0}, psf.real(f)).solve()
    im = psf.surfaceop(dom, {"lap": -0.3, "c": 1.0}, psf.imag(f)).solve()
    assert _relerr(u, re + 1j * im) < 1e-13


def test_dirichlet_data_on_open_surface_reproduces_harmonic_polynomial():
    g = lambda x, y, z: x**2 - y**2 + 3 * x * y
    xs, ys, zs = [], [], []
    for x0, x1, y0, y1 in [(-1, 0, -1, 0), (0, 1, -1, 0), (-1, 0, 0, 1), (0, 1, 0, 1)]:
        X, Y = chebpts2(8, 8, [x0, x1, y0, y1])
        xs.append(X)
        ys.append(Y)
        zs.append(0 * X)
    flat = psf.SurfaceMesh(xs, ys, zs)
    u = psf.surfaceop(flat, {"lap": 1.0}, 0.0).solve(bc=g)
    assert _relerr(u, psf.field(g, flat)) < 1e-12


def test_singular_quadrilateral_patch_is_exact_for_polynomials():
    # A quad with a 180-degree corner covers the triangle (0,0)-(1,0)-(0,1); J vanishes at a corner.
    u, v = chebpts2(8, 8, [0.0, 1.0, 0.0, 1.0])
    A, B, C, D = (np.array(p, float) for p in ([0, 0, 0], [0.5, 0, 0], [1, 0, 0], [0, 1, 0]))
    P = (
        A[:, None, None] * (1 - u) * (1 - v)
        + B[:, None, None] * u * (1 - v)
        + C[:, None, None] * u * v
        + D[:, None, None] * (1 - u) * v
    )
    dom = psf.SurfaceMesh([P[0]], [P[1]], [P[2]])
    assert dom.singular == [True]
    uh = psf.surfaceop(dom, {"lap": 1.0}, lambda x, y, z: -2.0 * (x + y)).solve()
    exact = psf.field(lambda x, y, z: x * y * (1 - x - y), dom)
    assert _relerr(uh, exact) < 1e-10


def test_snap_points_groups_nearby_points():
    pts = np.array([[0, 0, 0], [1e-12, 0, 0], [1, 1, 1], [1, 1, 1 + 1e-13], [2, 0, 0]], dtype=float)
    ids = hps.snap_points(pts, 1e-9)
    assert ids[0] == ids[1] and ids[2] == ids[3] and len({ids[0], ids[2], ids[4]}) == 3


def test_solver_stats_report_tree_shape():
    stats = psf.surfaceop(psf.sphere(6, 1), {"lap": -1.0, "c": 1.0}).stats()
    assert stats["npatches"] == 24 and stats["nmerges"] == 23
    assert stats["max_separator"] > 0 and stats["levels"] >= 5
