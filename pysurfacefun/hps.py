"""
Hierarchical Poincare--Steklov (HPS) fast direct solver engine.

This module is geometry agnostic.  The quadrilateral (:mod:`pysurfacefun.core`)
and triangular (:mod:`pysurfacefun.tri`) discretizations build dense *leaf*
solution operators for every surface patch and hand them to
:class:`HPSSolver`, which

1. identifies shared patch edges topologically by snapping edge end points and
   edge midpoints to integer ids (robust even when two different edges share
   both end points, as on coarse periodic meshes),
2. orders the merges by *nested dissection* of the patch adjacency graph:
   recursive principal-axis bisection of the patch centroids, refined by
   Fiduccia--Mattheyses passes that minimize the number of interface
   unknowns while keeping the halves balanced,
3. eliminates the interface unknowns bottom-up with dense Schur complements
   (the factorization, which is independent of the right-hand side), and
4. compiles the factored tree into level-by-level batched gathers and matrix
   products, so a solve costs a handful of NumPy calls per tree level
   instead of Python recursion over every node.

Notation
--------
Every node of the merge tree owns a list of boundary *edges*.  An edge carries
``size`` skeleton unknowns; the node's boundary unknowns are the concatenation
of its edges' unknowns.  ``D2N`` maps Dirichlet data on the boundary unknowns
to (possibly scaled) outward fluxes.  Merging two children eliminates the
unknowns on their shared edges (the *separator*)::

    u_sep = S @ bc + A_inv @ z_part,      flux = D2N @ bc + du_part.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

Array = np.ndarray
MergeTree = int | tuple
"""A merge tree: a patch index (leaf) or a pair ``(left, right)`` of subtrees."""


# ---------------------------------------------------------------------------
# Tree nodes.
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Patch:
    """Node of an HPS merge tree.

    Attributes
    ----------
    ids:
        Patch indices covered by this node, in tree order.
    edge_keys:
        ``(ne, 3)`` integer ids ``(start, end, midpoint)`` of the boundary edges.
    edge_sizes:
        ``(ne,)`` number of skeleton unknowns on each boundary edge.
    D2N:
        Dirichlet-to-Neumann map on the boundary unknowns.  Released once the
        node has been merged into its parent (the root keeps it).
    xyz:
        ``(b, 3)`` coordinates of the boundary unknowns.
    w:
        ``(b,)`` surface quadrature weights of the boundary unknowns.
    scale:
        ``(b,)`` flux scaling (``None`` means one).  Patches with a degenerate
        parametrization multiply their equations by powers of the Jacobian.
    """

    ids: list[int]
    edge_keys: Array
    edge_sizes: Array
    D2N: Array | None
    xyz: Array
    w: Array
    scale: Array | None

    @property
    def nboundary(self) -> int:
        """Number of boundary (skeleton) unknowns."""
        return int(self.xyz.shape[0])


@dataclass(eq=False)
class Leaf(Patch):
    """Leaf node: the dense solution operator of one surface patch."""

    index: int = 0
    S: Array | None = None
    height: int = 0


@dataclass(eq=False)
class Parent(Patch):
    """Internal node produced by merging two children.

    ``ext1``/``sep1`` index the exterior/separator unknowns of ``child1``;
    ``ext2``/``sep2`` those of ``child2`` with ``sep2`` permuted into the
    orientation of ``child1`` so that ``sep1[i]`` and ``sep2[i]`` are the same
    physical point.
    """

    child1: Patch | None = None
    child2: Patch | None = None
    ext1: Array | None = None
    sep1: Array | None = None
    ext2: Array | None = None
    sep2: Array | None = None
    A_inv: Array | None = None
    S: Array | None = None
    M: Array | None = None
    rankdef_merged: bool = False
    height: int = 1

    @property
    def nseparator(self) -> int:
        """Number of interface unknowns eliminated by this merge."""
        return 0 if self.sep1 is None else int(self.sep1.size)


@dataclass(eq=False)
class LeafOperators:
    """Stacked dense leaf operators for all ``P`` patches of a mesh.

    Shapes use ``N`` nodes per patch, ``m`` interior nodes, ``b`` skeleton
    unknowns, and ``ne`` edges per patch.

    Attributes
    ----------
    S:
        ``(P, N, b)`` homogeneous solution operator: skeleton Dirichlet data to
        nodal values.
    D2N:
        ``(P, b, b)`` Dirichlet-to-Neumann maps.
    Ainv:
        ``(P, m, m)`` interior solve applied to the right-hand side (any
        right-hand-side scaling is already folded in).
    G:
        ``(P, b, m)`` right-hand side to outward flux of the particular solution.
    interior:
        ``(m,)`` interior node indices.
    edge_sizes:
        ``(ne,)`` skeleton unknowns per edge.
    edge_points:
        ``(P, ne, 3, 3)`` start point, end point, and midpoint of every edge.
    xyz, w:
        ``(P, b, 3)`` skeleton points and ``(P, b)`` quadrature weights.
    scale:
        ``(P, b)`` flux scaling or ``None``.
    centroids:
        ``(P, 3)`` patch centroids used for geometric bisection.
    """

    S: Array
    D2N: Array
    Ainv: Array
    G: Array
    interior: Array
    edge_sizes: Array
    edge_points: Array
    xyz: Array
    w: Array
    scale: Array | None
    centroids: Array

    @property
    def npatches(self) -> int:
        return int(self.S.shape[0])

    @property
    def nnodes(self) -> int:
        return int(self.S.shape[1])


# ---------------------------------------------------------------------------
# Topology: point snapping, edge keys, and patch adjacency.
# ---------------------------------------------------------------------------


def snap_points(points: Array, tol: float) -> Array:
    """Assign integer ids so that points within ``tol`` (max norm) share an id.

    Points are sorted along a generic direction; only points whose projections
    are within the tolerance window are compared, so the cost is essentially
    ``O(N log N)``.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    npts = points.shape[0]
    ids = np.full(npts, -1, dtype=np.int64)
    if npts == 0:
        return ids

    direction = np.array([0.7331, 0.4659, 0.4955])
    direction /= np.linalg.norm(direction)
    proj = points @ direction
    order = np.argsort(proj, kind="stable")
    sorted_proj = proj[order]
    window = 2.0 * tol

    # Runs of consecutive sorted points closer than the window; points in
    # different runs cannot be within ``tol`` of each other.
    breaks = np.flatnonzero(np.diff(sorted_proj) > window) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [npts]))
    lengths = stops - starts

    singles = starts[lengths == 1]
    next_id = 0
    ids[order[singles]] = np.arange(singles.size)
    next_id = singles.size

    for start, stop in zip(starts[lengths > 1], stops[lengths > 1]):
        members = order[start:stop]
        for pos, i in enumerate(members):
            if ids[i] >= 0:
                continue
            ids[i] = next_id
            rest = members[pos + 1 :]
            rest = rest[ids[rest] < 0]
            if rest.size:
                close = np.max(np.abs(points[rest] - points[i]), axis=1) <= tol
                ids[rest[close]] = next_id
            next_id += 1
    return ids


def edge_keys_from_points(edge_points: Array, tol: float | None = None) -> Array:
    """Integer edge keys ``(start, end, tag)`` for ``(P, ne, 3, 3)`` edge points.

    ``edge_points[p, e]`` holds the start point, end point, and midpoint of
    edge ``e`` of patch ``p``.  End points are snapped to vertex ids with
    tolerance ``tol``.  Edges are then identified by their unordered end-point
    ids; only when more than two edges share the same end points (for example
    the two halves of a circle on a coarse periodic mesh) are they told apart
    by their midpoints, which get a nonzero ``tag``.
    """
    edge_points = np.asarray(edge_points, dtype=float)
    npatches, nedges = edge_points.shape[:2]
    scale = float(np.max(np.abs(edge_points))) if edge_points.size else 1.0
    scale = max(scale, 1.0)
    if tol is None:
        tol = 1.0e-8 * scale
    ends = snap_points(edge_points[:, :, :2, :].reshape(-1, 3), tol).reshape(npatches, nedges, 2)
    keys = np.zeros((npatches, nedges, 3), dtype=np.int64)
    keys[:, :, :2] = ends

    flat_lo = np.minimum(ends[..., 0], ends[..., 1]).ravel()
    flat_hi = np.maximum(ends[..., 0], ends[..., 1]).ravel()
    order = np.lexsort((flat_hi, flat_lo))
    pairs = np.stack((flat_lo[order], flat_hi[order]), axis=1)
    breaks = np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [order.size]))
    mids = edge_points[:, :, 2, :].reshape(-1, 3)
    lengths = np.linalg.norm(edge_points[:, :, 1, :] - edge_points[:, :, 0, :], axis=-1).ravel()
    tags = keys[:, :, 2].reshape(-1)
    for start, stop in zip(starts, stops):
        if stop - start <= 2:
            continue
        members = order[start:stop]
        mid_tol = max(1.0e-6 * scale, 1.0e-3 * float(np.max(lengths[members])))
        tags[members] = snap_points(mids[members], mid_tol) + 1
    keys[:, :, 2] = tags.reshape(npatches, nedges)
    return keys


def _unordered_keys(edge_keys: Array) -> Array:
    lo = np.minimum(edge_keys[..., 0], edge_keys[..., 1])
    hi = np.maximum(edge_keys[..., 0], edge_keys[..., 1])
    return np.stack((lo, hi, edge_keys[..., 2]), axis=-1)


def patch_adjacency(edge_keys: Array, edge_sizes: Array) -> tuple[Array, Array, Array]:
    """Weighted patch adjacency graph in CSR form ``(indptr, indices, weights)``.

    Two patches are adjacent when they share an edge; the weight is the number
    of skeleton unknowns on the shared edges.
    """
    npatches, nedges = edge_keys.shape[:2]
    keys = _unordered_keys(edge_keys).reshape(-1, 3)
    owners = np.repeat(np.arange(npatches), nedges)
    sizes = np.tile(np.asarray(edge_sizes), npatches)

    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    same = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    rows: list[Array] = []
    cols: list[Array] = []
    vals: list[Array] = []
    # Pairs of consecutive equal keys (manifold edges); longer runs of
    # non-manifold edges are chained, which keeps the graph connected.
    first = order[:-1][same]
    second = order[1:][same]
    keep = owners[first] != owners[second]
    first, second = first[keep], second[keep]
    rows.extend((owners[first], owners[second]))
    cols.extend((owners[second], owners[first]))
    vals.extend((sizes[first], sizes[first]))

    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals).astype(float)
    else:
        r = c = np.zeros(0, dtype=np.int64)
        v = np.zeros(0)

    # Sum duplicate pairs (patches sharing several edges).
    flat = r * npatches + c
    uniq, inverse = np.unique(flat, return_inverse=True)
    weights = np.bincount(inverse, weights=v, minlength=uniq.size) if uniq.size else np.zeros(0)
    r = uniq // npatches
    c = uniq % npatches
    indptr = np.zeros(npatches + 1, dtype=np.int64)
    np.add.at(indptr, r + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, c.astype(np.int64), weights


def orientation_signs(edge_keys: Array, ccw: Array) -> Array:
    """Per-patch signs that make patch parametrizations consistently oriented.

    ``ccw[e]`` is ``+1`` when edge ``e`` of the reference patch is traversed in
    its stored (start to end) direction by the counterclockwise boundary of the
    reference domain and ``-1`` otherwise.  Two consistently oriented patches
    traverse a shared edge in opposite directions, so a breadth-first sweep
    over shared edges fixes the relative sign of every connected component
    (the first patch of each component keeps ``+1``).  Collapsed edges are
    ignored; on non-orientable surfaces the first assignment wins.
    """
    edge_keys = np.asarray(edge_keys)
    ccw = np.asarray(ccw)
    npatches, nedges = edge_keys.shape[:2]
    unordered = _unordered_keys(edge_keys)
    owners: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for p in range(npatches):
        for e in range(nedges):
            owners.setdefault(tuple(unordered[p, e].tolist()), []).append((p, e))
    signs = np.zeros(npatches, dtype=np.int64)
    for seed in range(npatches):
        if signs[seed]:
            continue
        signs[seed] = 1
        stack = [seed]
        while stack:
            p = stack.pop()
            for e in range(nedges):
                start, end = int(edge_keys[p, e, 0]), int(edge_keys[p, e, 1])
                if start == end:
                    continue
                along_p = ccw[e] * (1 if start < end else -1)
                for q, f in owners[tuple(unordered[p, e].tolist())]:
                    if q == p or signs[q]:
                        continue
                    along_q = ccw[f] * (1 if edge_keys[q, f, 0] < edge_keys[q, f, 1] else -1)
                    signs[q] = -signs[p] * along_p * along_q
                    stack.append(q)
    return signs


# ---------------------------------------------------------------------------
# Merge trees.
# ---------------------------------------------------------------------------


def natural_tree(npatches: int) -> MergeTree:
    """Balanced tree merging consecutive patch indices (the legacy ordering)."""
    if npatches < 1:
        raise ValueError("a merge tree needs at least one patch")
    nodes: list[MergeTree] = list(range(npatches))
    while len(nodes) > 1:
        paired: list[MergeTree] = []
        for k in range(0, len(nodes), 2):
            paired.append((nodes[k], nodes[k + 1]) if k + 1 < len(nodes) else nodes[k])
        nodes = paired
    return nodes[0]


def _concat_ranges(starts: Array, counts: Array) -> Array:
    total = int(np.sum(counts))
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    shifts = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    return shifts + np.arange(total)


def _subgraph(nodes: Array, adjacency: tuple[Array, Array, Array], local: Array):
    indptr, indices, weights = adjacency
    local[nodes] = np.arange(nodes.size)
    counts = indptr[nodes + 1] - indptr[nodes]
    pos = _concat_ranges(indptr[nodes], counts)
    rows = np.repeat(np.arange(nodes.size), counts)
    cols = local[indices[pos]]
    keep = cols >= 0
    rows, cols, vals = rows[keep], cols[keep], weights[pos][keep]
    local[nodes] = -1
    sub_indptr = np.zeros(nodes.size + 1, dtype=np.int64)
    np.add.at(sub_indptr, rows + 1, 1)
    return np.cumsum(sub_indptr), cols, vals


def _fm_refine(sub: tuple[Array, Array, Array], side: Array, lo: int, hi: int, max_passes: int = 8) -> Array:
    """Fiduccia--Mattheyses refinement of a bisection (``side`` in {0, 1})."""
    indptr, cols, vals = sub
    n = side.size
    if cols.size == 0:
        return side
    rows = np.repeat(np.arange(n), np.diff(indptr))
    nbrs = [cols[indptr[u] : indptr[u + 1]].tolist() for u in range(n)]
    wts = [vals[indptr[u] : indptr[u + 1]].tolist() for u in range(n)]
    side = side.copy()

    for _ in range(max_passes):
        cross = side[rows] != side[cols]
        gain = np.bincount(rows, weights=np.where(cross, vals, -vals), minlength=n)
        size0 = int(np.sum(side == 0))
        boundary = np.flatnonzero(np.bincount(rows, weights=cross, minlength=n) > 0)
        heap = [(-gain[u], int(u)) for u in boundary]
        heapq.heapify(heap)
        gain_list = gain.tolist()
        side_list = side.tolist()
        locked = [False] * n
        moves: list[int] = []
        total = 0.0
        best = 0.0
        best_len = 0
        stall_limit = max(25, n // 20)
        while heap:
            neg_gain, u = heapq.heappop(heap)
            if locked[u] or -neg_gain != gain_list[u]:
                continue
            new_size0 = size0 - 1 if side_list[u] == 0 else size0 + 1
            if not lo <= new_size0 <= hi:
                continue
            side_list[u] ^= 1
            size0 = new_size0
            locked[u] = True
            total += gain_list[u]
            moves.append(u)
            su = side_list[u]
            for v, w in zip(nbrs[u], wts[u]):
                if not locked[v]:
                    gain_list[v] += -2.0 * w if side_list[v] == su else 2.0 * w
                    heapq.heappush(heap, (-gain_list[v], v))
            if total > best + 1e-12:
                best = total
                best_len = len(moves)
            elif len(moves) - best_len > stall_limit:
                break
        for u in moves[best_len:]:
            side_list[u] ^= 1
        side = np.asarray(side_list)
        if best <= 1e-12:
            break
    return side


def nested_dissection_tree(
    centroids: Array,
    adjacency: tuple[Array, Array, Array],
    imbalance: float = 0.2,
    refine: bool = True,
) -> MergeTree:
    """Nested-dissection merge tree for a patch adjacency graph.

    Each set of patches is split in two by the median of the centroids along
    the principal axis; the cut is then refined by Fiduccia--Mattheyses passes
    that reduce the number of shared skeleton unknowns (the size of the dense
    interface system eliminated when the halves are merged) while keeping each
    half within ``(1 +/- imbalance) / 2`` of the set size.
    """
    centroids = np.asarray(centroids, dtype=float)
    npatches = centroids.shape[0]
    if npatches < 1:
        raise ValueError("a merge tree needs at least one patch")
    local = np.full(npatches, -1, dtype=np.int64)

    def split(nodes: Array) -> tuple[Array, Array]:
        size = nodes.size
        pts = centroids[nodes] - centroids[nodes].mean(axis=0)
        cov = pts.T @ pts
        _, vecs = np.linalg.eigh(cov)
        proj = pts @ vecs[:, -1]
        order = np.argsort(proj, kind="stable")
        half = (size + 1) // 2
        side = np.ones(size, dtype=np.int64)
        side[order[:half]] = 0
        if refine and size > 2:
            lo = max(1, int(np.floor(0.5 * size * (1.0 - imbalance))))
            hi = min(size - 1, int(np.ceil(0.5 * size * (1.0 + imbalance))))
            side = _fm_refine(_subgraph(nodes, adjacency, local), side, lo, hi)
        left = nodes[side == 0]
        right = nodes[side == 1]
        if left.size == 0 or right.size == 0:
            left, right = nodes[order[:half]], nodes[order[half:]]
        return left, right

    def build(nodes: Array) -> MergeTree:
        if nodes.size == 1:
            return int(nodes[0])
        left, right = split(nodes)
        return (build(left), build(right))

    return build(np.arange(npatches))


def matching_tree(npatches: int, adjacency: tuple[Array, Array, Array]) -> MergeTree:
    """Bottom-up tree that greedily merges adjacent clusters (heavy-edge matching)."""
    if npatches < 1:
        raise ValueError("a merge tree needs at least one patch")
    indptr, indices, weights = adjacency
    trees: list[MergeTree] = list(range(npatches))
    cluster = np.arange(npatches)
    ncl = npatches
    while ncl > 1:
        # Cluster graph weights.
        rows = np.repeat(np.arange(npatches), np.diff(indptr))
        cr, cc = cluster[rows], cluster[indices]
        keep = cr != cc
        pair_w: dict[tuple[int, int], float] = {}
        for a, b, w in zip(cr[keep].tolist(), cc[keep].tolist(), weights[keep].tolist()):
            pair_w[(a, b)] = pair_w.get((a, b), 0.0) + w
        matched = [False] * ncl
        new_index = [-1] * ncl
        new_trees: list[MergeTree] = []
        for (a, b), _ in sorted(pair_w.items(), key=lambda item: (-item[1], item[0])):
            if a < b and not matched[a] and not matched[b]:
                matched[a] = matched[b] = True
                new_index[a] = new_index[b] = len(new_trees)
                new_trees.append((trees[a], trees[b]))
        if not new_trees:
            # Disconnected clusters: pair them in order.
            for a in range(0, ncl - 1, 2):
                matched[a] = matched[a + 1] = True
                new_index[a] = new_index[a + 1] = len(new_trees)
                new_trees.append((trees[a], trees[a + 1]))
        for a in range(ncl):
            if not matched[a]:
                new_index[a] = len(new_trees)
                new_trees.append(trees[a])
        cluster = np.asarray(new_index)[cluster]
        trees = new_trees
        ncl = len(trees)
    return trees[0]


def tree_from_merge_idx(merge_idx: list[list[tuple[int, int | None]]], npatches: int) -> MergeTree:
    """Convert a legacy level-by-level merge schedule into a merge tree."""
    nodes: list[MergeTree] = list(range(npatches))
    for level in merge_idx:
        nodes = [nodes[a] if b is None else (nodes[a], nodes[b]) for a, b in level]
    if len(nodes) != 1:
        raise ValueError("merge schedule does not reduce to a single root")
    tree = nodes[0]
    validate_tree(tree, npatches)
    return tree


def tree_table(tree: MergeTree, npatches: int) -> tuple[int, list[tuple[int, int]]]:
    """Flatten a merge tree into ``(root, children)``.

    Leaves keep their patch index as node id; internal node ``npatches + i``
    merges ``children[i] = (left, right)``.  Children always precede their
    parents (post-order), and the traversal is iterative so arbitrarily deep
    user-supplied trees are fine.
    """
    children: list[tuple[int, int]] = []
    values: list[int] = []
    stack: list[tuple[MergeTree, bool]] = [(tree, False)]
    while stack:
        node, expanded = stack.pop()
        if not isinstance(node, tuple):
            values.append(int(node))
        elif expanded:
            right = values.pop()
            left = values.pop()
            children.append((left, right))
            values.append(npatches + len(children) - 1)
        else:
            if len(node) != 2:
                raise ValueError("merge tree nodes must be pairs")
            stack.append((node, True))
            stack.append((node[1], False))
            stack.append((node[0], False))
    return values[0], children


def tree_leaves(tree: MergeTree) -> list[int]:
    """Patch indices of a merge tree in left-to-right order."""
    out: list[int] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, tuple):
            stack.append(node[1])
            stack.append(node[0])
        else:
            out.append(int(node))
    return out


def validate_tree(tree: MergeTree, npatches: int) -> None:
    """Raise ``ValueError`` unless every patch appears exactly once."""
    if sorted(tree_leaves(tree)) != list(range(npatches)):
        raise ValueError("merge tree must contain every patch index exactly once")


def merge_idx_from_tree(tree: MergeTree, npatches: int) -> list[list[tuple[int, int | None]]]:
    """Level-by-level merge schedule (the legacy ``merge_idx`` format) of a tree.

    Level ``h`` performs every merge whose subtree height is ``h``; entries
    that do not take part in a merge are carried over as ``(index, None)``.
    """
    _, children = tree_table(tree, npatches)
    nnodes = npatches + len(children)
    height = np.zeros(nnodes, dtype=np.int64)
    parent = np.full(nnodes, -1, dtype=np.int64)
    for i, (left, right) in enumerate(children):
        node = npatches + i
        height[node] = 1 + max(height[left], height[right])
        parent[left] = parent[right] = node

    schedule: list[list[tuple[int, int | None]]] = []
    current = list(range(npatches))
    for level in range(1, int(height.max(initial=0)) + 1):
        position = {node: i for i, node in enumerate(current)}
        emitted: set[int] = set()
        pairs: list[tuple[int, int | None]] = []
        new_nodes: list[int] = []
        for i, node in enumerate(current):
            par = int(parent[node])
            if par >= 0 and height[par] == level:
                if par not in emitted:
                    emitted.add(par)
                    left, right = children[par - npatches]
                    pairs.append((position[left], position[right]))
                    new_nodes.append(par)
            else:
                pairs.append((i, None))
                new_nodes.append(node)
        schedule.append(pairs)
        current = new_nodes
    return schedule


# ---------------------------------------------------------------------------
# Merging.
# ---------------------------------------------------------------------------


def _edge_offsets(sizes: Array) -> Array:
    return np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)


def match_edges(a: Patch, b: Patch) -> tuple[list[int], list[int], list[bool]]:
    """Shared edges of two nodes: indices into ``a``, into ``b``, and whether reversed."""
    lookup: dict[tuple[int, int, int], list[int]] = {}
    for j, (s, e, m) in enumerate(b.edge_keys.tolist()):
        lookup.setdefault((min(s, e), max(s, e), m), []).append(j)
    ia: list[int] = []
    ib: list[int] = []
    flipped: list[bool] = []
    used: set[int] = set()
    for i, (s, e, m) in enumerate(a.edge_keys.tolist()):
        for j in lookup.get((min(s, e), max(s, e), m), ()):
            if j not in used:
                used.add(j)
                ia.append(i)
                ib.append(j)
                flipped.append(int(b.edge_keys[j, 0]) != s)
                break
    return ia, ib, flipped


def merge_patches(a: Patch, b: Patch, rankdef: bool = False) -> Parent:
    """Merge two HPS nodes by eliminating the unknowns on their shared edges.

    With ``rankdef=True`` and a closed result (no exterior unknowns), the
    rank-one term ``w w^T`` regularizes the constant null space of a pure
    Laplace--Beltrami operator.
    """
    ia, ib, flipped = match_edges(a, b)
    offa = _edge_offsets(a.edge_sizes)
    offb = _edge_offsets(b.edge_sizes)
    for i, j in zip(ia, ib):
        if a.edge_sizes[i] != b.edge_sizes[j]:
            raise ValueError("shared edges must carry the same number of skeleton unknowns")

    sep1 = np.concatenate([np.arange(offa[i], offa[i + 1]) for i in ia]) if ia else np.zeros(0, dtype=np.int64)
    sep2_parts = []
    for j, flip in zip(ib, flipped):
        rng = np.arange(offb[j], offb[j + 1])
        sep2_parts.append(rng[::-1] if flip else rng)
    sep2 = np.concatenate(sep2_parts) if sep2_parts else np.zeros(0, dtype=np.int64)

    mask_a = np.ones(a.nboundary, dtype=bool)
    mask_a[sep1] = False
    mask_b = np.ones(b.nboundary, dtype=bool)
    mask_b[sep2] = False
    ext1 = np.flatnonzero(mask_a)
    ext2 = np.flatnonzero(mask_b)

    D2Na = a.D2N
    D2Nb = b.D2N
    assert D2Na is not None and D2Nb is not None
    scale_a = None if a.scale is None else a.scale[sep1]
    scale_b = None if b.scale is None else b.scale[sep2]

    Aaa = D2Na[np.ix_(sep1, sep1)]
    Abb = D2Nb[np.ix_(sep2, sep2)]
    Za = D2Na[np.ix_(sep1, ext1)]
    Zb = D2Nb[np.ix_(sep2, ext2)]
    if scale_b is not None:
        Aaa = scale_b[:, None] * Aaa
        Za = scale_b[:, None] * Za
    if scale_a is not None:
        Abb = scale_a[:, None] * Abb
        Zb = scale_a[:, None] * Zb
    A = -(Aaa + Abb)
    Z = np.hstack((Za, Zb))

    closed = ext1.size == 0 and ext2.size == 0
    rankdef_merged = bool(rankdef and closed and A.size)
    if rankdef_merged:
        wts = a.w[sep1].reshape(-1, 1)
        A = A + wts @ wts.T

    A_inv = np.linalg.inv(A) if A.size else np.zeros((0, 0), dtype=A.dtype)
    S = A_inv @ Z
    M = np.vstack((D2Na[np.ix_(ext1, sep1)], D2Nb[np.ix_(ext2, sep2)]))
    nb = ext1.size + ext2.size
    D2N = M @ S if sep1.size else np.zeros((nb, nb), dtype=np.result_type(D2Na, D2Nb))
    D2N[: ext1.size, : ext1.size] += D2Na[np.ix_(ext1, ext1)]
    D2N[ext1.size :, ext1.size :] += D2Nb[np.ix_(ext2, ext2)]

    keep_a = np.ones(len(a.edge_sizes), dtype=bool)
    keep_a[ia] = False
    keep_b = np.ones(len(b.edge_sizes), dtype=bool)
    keep_b[ib] = False
    edge_keys = np.vstack((a.edge_keys[keep_a], b.edge_keys[keep_b]))
    edge_sizes = np.concatenate((a.edge_sizes[keep_a], b.edge_sizes[keep_b]))

    if a.scale is None and b.scale is None:
        scale = None
    else:
        sa = np.ones(a.nboundary) if a.scale is None else a.scale
        sb = np.ones(b.nboundary) if b.scale is None else b.scale
        scale = np.concatenate((sa[ext1], sb[ext2]))

    height = 1 + max(getattr(a, "height", 0), getattr(b, "height", 0))
    return Parent(
        ids=a.ids + b.ids,
        edge_keys=edge_keys,
        edge_sizes=edge_sizes,
        D2N=D2N,
        xyz=np.vstack((a.xyz[ext1], b.xyz[ext2])),
        w=np.concatenate((a.w[ext1], b.w[ext2])),
        scale=scale,
        child1=a,
        child2=b,
        ext1=ext1,
        sep1=sep1,
        ext2=ext2,
        sep2=sep2,
        A_inv=A_inv,
        S=S,
        M=M,
        rankdef_merged=rankdef_merged,
        height=height,
    )


# ---------------------------------------------------------------------------
# Compiled solves.
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _Group:
    """Parents at one tree height that share the same (separator, boundary) sizes."""

    k: int
    b: int
    count: int
    ustart: int
    dstart: int
    idx_sep1: Array
    idx_sep2: Array
    idx_ext: Array
    scale_a: Array | None
    scale_b: Array | None
    A_inv: Array
    S: Array
    M: Array


class _CompiledTree:
    """Flat, level-batched representation of a factored HPS tree."""

    def __init__(self, root: Patch, leaves: list[Leaf], leaf_ops: LeafOperators):
        self.npatches = len(leaves)
        self.leaf_b = int(leaf_ops.S.shape[2])
        self.S_leaf = leaf_ops.S
        self.Ainv_leaf = leaf_ops.Ainv
        self.G_leaf = leaf_ops.G
        self.interior = leaf_ops.interior
        self.nnodes = leaf_ops.nnodes

        parents: list[Parent] = []
        stack: list[Patch] = [root]
        while stack:
            node = stack.pop()
            if isinstance(node, Parent):
                parents.append(node)
                stack.append(node.child1)  # type: ignore[arg-type]
                stack.append(node.child2)  # type: ignore[arg-type]

        doff: dict[int, int] = {}
        for leaf in leaves:
            doff[id(leaf)] = leaf.index * self.leaf_b
        dnext = self.npatches * self.leaf_b

        by_level: dict[int, dict[tuple[int, int], list[Parent]]] = {}
        for node in parents:
            key = (node.nseparator, node.nboundary)
            by_level.setdefault(node.height, {}).setdefault(key, []).append(node)

        self.levels: list[list[_Group]] = []
        unext = 0
        for height in sorted(by_level):
            groups: list[_Group] = []
            for (k, b), members in sorted(by_level[height].items()):
                dstart = dnext
                for node in members:
                    doff[id(node)] = dnext
                    dnext += b
                groups.append(self._make_group(members, k, b, unext, dstart, doff))
                unext += k * len(members)
            self.levels.append(groups)
        self.nd = dnext
        self.nu = unext
        self.root_offset = doff[id(root)]
        self.root_b = root.nboundary
        self.dtype = np.result_type(
            leaf_ops.S, leaf_ops.Ainv, leaf_ops.G, *(g.A_inv for lvl in self.levels for g in lvl)
        )

    @staticmethod
    def _make_group(members: list[Parent], k: int, b: int, ustart: int, dstart: int, doff: dict[int, int]) -> _Group:
        idx_sep1 = np.concatenate([doff[id(p.child1)] + p.sep1 for p in members]) if k else np.zeros(0, dtype=np.int64)
        idx_sep2 = np.concatenate([doff[id(p.child2)] + p.sep2 for p in members]) if k else np.zeros(0, dtype=np.int64)
        idx_ext = (
            np.concatenate(
                [np.concatenate((doff[id(p.child1)] + p.ext1, doff[id(p.child2)] + p.ext2)) for p in members]
            )
            if b
            else np.zeros(0, dtype=np.int64)
        )
        scale_a = scale_b = None
        if any(p.child1.scale is not None or p.child2.scale is not None for p in members):
            sa_parts = []
            sb_parts = []
            for p in members:
                c1, c2 = p.child1, p.child2
                sa_parts.append(np.ones(p.nseparator) if c1.scale is None else c1.scale[p.sep1])
                sb_parts.append(np.ones(p.nseparator) if c2.scale is None else c2.scale[p.sep2])
            scale_a = np.concatenate(sa_parts).reshape(-1, 1) if k else None
            scale_b = np.concatenate(sb_parts).reshape(-1, 1) if k else None

        # Stack the dense blocks and point the tree nodes at views of the stacks.
        A_inv = np.stack([p.A_inv for p in members])
        S = np.stack([p.S for p in members])
        M = np.stack([p.M for p in members])
        for i, p in enumerate(members):
            p.A_inv = A_inv[i]
            p.S = S[i]
            p.M = M[i]
        return _Group(k, b, len(members), ustart, dstart, idx_sep1, idx_sep2, idx_ext, scale_a, scale_b, A_inv, S, M)

    def solve(self, F: Array, bc_root: Array | None = None) -> Array:
        """Solve for interior right-hand sides ``F`` of shape ``(P, m, r)``.

        Returns nodal values of shape ``(P, N, r)``.
        """
        P, _, r = F.shape
        dtype = np.result_type(self.dtype, F)
        DU = np.empty((self.nd, r), dtype=dtype)
        UP = np.empty((self.nu, r), dtype=dtype)

        # Upward sweep: particular solutions and their outward fluxes.
        UINT = np.matmul(self.Ainv_leaf, F)
        DU[: P * self.leaf_b] = np.matmul(self.G_leaf, F).reshape(-1, r)
        for groups in self.levels:
            for g in groups:
                if g.k:
                    Z = DU[g.idx_sep1]
                    Zb = DU[g.idx_sep2]
                    if g.scale_b is not None:
                        Z = Z * g.scale_b
                        Zb = Zb * g.scale_a
                    Z += Zb
                    up = np.matmul(g.A_inv, Z.reshape(g.count, g.k, r))
                    UP[g.ustart : g.ustart + g.count * g.k] = up.reshape(-1, r)
                if g.b:
                    ext = DU[g.idx_ext].reshape(g.count, g.b, r)
                    if g.k:
                        ext += np.matmul(g.M, up)
                    DU[g.dstart : g.dstart + g.count * g.b] = ext.reshape(-1, r)

        # Downward sweep: interface values from the root to the leaves.
        BC = DU  # reuse storage: every DU entry is consumed before BC overwrites it
        BC[self.root_offset : self.root_offset + self.root_b] = 0.0 if bc_root is None else bc_root
        for groups in reversed(self.levels):
            for g in groups:
                if g.k:
                    uif = UP[g.ustart : g.ustart + g.count * g.k].reshape(g.count, g.k, r)
                    if g.b:
                        bcg = BC[g.dstart : g.dstart + g.count * g.b].reshape(g.count, g.b, r)
                        uif = uif + np.matmul(g.S, bcg)
                    flat = uif.reshape(-1, r)
                if g.b:
                    BC[g.idx_ext] = BC[g.dstart : g.dstart + g.count * g.b]
                if g.k:
                    BC[g.idx_sep1] = flat
                    BC[g.idx_sep2] = flat

        bc_leaf = BC[: P * self.leaf_b].reshape(P, self.leaf_b, r)
        U = np.matmul(self.S_leaf, bc_leaf)
        U[:, self.interior, :] += UINT
        return U


def _as_real_view(F: Array) -> tuple[Array, bool]:
    """View a complex ``(..., r)`` array as a real ``(..., 2r)`` array."""
    if np.iscomplexobj(F):
        F = np.ascontiguousarray(F)
        return F.view(np.float64 if F.dtype == np.complex128 else np.float32), True
    return F, False


class HPSSolver:
    """Factor and solve an HPS system defined by stacked leaf operators.

    Parameters
    ----------
    leaf_ops:
        Dense leaf operators for every patch.
    tree:
        Merge tree (``None`` builds a nested-dissection tree).
    strategy:
        ``"nested_dissection"`` (default), ``"natural"`` (merge consecutive
        indices), or ``"matching"`` (greedy adjacent pairing).
    """

    def __init__(
        self,
        leaf_ops: LeafOperators,
        tree: MergeTree | None = None,
        strategy: str = "nested_dissection",
        lazy_tree: bool = False,
    ):
        self.leaf_ops = leaf_ops
        self.strategy = strategy
        self.edge_keys = edge_keys_from_points(leaf_ops.edge_points)
        self.leaves = [
            Leaf(
                ids=[k],
                edge_keys=self.edge_keys[k],
                edge_sizes=np.asarray(leaf_ops.edge_sizes, dtype=np.int64),
                D2N=leaf_ops.D2N[k],
                xyz=leaf_ops.xyz[k],
                w=leaf_ops.w[k],
                scale=None if leaf_ops.scale is None else leaf_ops.scale[k],
                index=k,
                S=leaf_ops.S[k],
            )
            for k in range(leaf_ops.npatches)
        ]
        self._adjacency: tuple[Array, Array, Array] | None = None
        self._tree: MergeTree | None = None
        self.root: Patch | None = None
        self._compiled: _CompiledTree | None = None
        if tree is not None:
            self.tree = tree
        elif not lazy_tree:
            self._tree = self.default_tree(strategy)

    @property
    def tree(self) -> MergeTree:
        """Merge tree (computed from :attr:`strategy` on first access)."""
        if self._tree is None:
            self._tree = self.default_tree(self.strategy)
        return self._tree

    @tree.setter
    def tree(self, tree: MergeTree) -> None:
        if self.root is not None:
            raise RuntimeError("the merge tree cannot change after factor()")
        validate_tree(tree, self.leaf_ops.npatches)
        self._tree = tree

    @property
    def adjacency(self) -> tuple[Array, Array, Array]:
        """Weighted patch adjacency graph ``(indptr, indices, weights)``."""
        if self._adjacency is None:
            self._adjacency = patch_adjacency(self.edge_keys, self.leaf_ops.edge_sizes)
        return self._adjacency

    def default_tree(self, strategy: str = "nested_dissection") -> MergeTree:
        """Merge tree for a named strategy."""
        strategy = strategy.lower().replace("-", "_")
        npatches = self.leaf_ops.npatches
        if strategy in {"nested_dissection", "nd", "default", "auto"}:
            if npatches == 1:
                return 0
            return nested_dissection_tree(self.leaf_ops.centroids, self.adjacency)
        if strategy in {"natural", "sequential", "legacy"}:
            return natural_tree(npatches)
        if strategy in {"matching", "adjacent", "greedy"}:
            return matching_tree(npatches, self.adjacency)
        raise ValueError("strategy must be 'nested_dissection', 'natural', or 'matching'")

    @property
    def factored(self) -> bool:
        return self.root is not None

    def factor(self, rankdef: bool = False) -> Patch:
        """Eliminate all interface unknowns bottom-up and compile the solver.

        With ``rankdef=True`` the root merge adds a rank-one term that removes
        the constant null space of a pure Laplace--Beltrami operator on a
        closed surface.
        """
        npatches = self.leaf_ops.npatches
        root_id, children = tree_table(self.tree, npatches)
        nodes: list[Patch | None] = list(self.leaves) + [None] * len(children)
        for i, (left, right) in enumerate(children):
            a, b = nodes[left], nodes[right]
            assert a is not None and b is not None
            nodes[npatches + i] = merge_patches(a, b, rankdef=rankdef and npatches + i == root_id)
            # Dirichlet-to-Neumann maps are only needed by a node's parent.
            a.D2N = None
            b.D2N = None
        root = nodes[root_id]
        assert root is not None
        self.leaf_ops.D2N = None  # type: ignore[assignment]
        self.root = root
        self._compiled = _CompiledTree(root, self.leaves, self.leaf_ops)
        return root

    def solve(self, F: Array, bc: Array | None = None) -> Array:
        """Solve for interior right-hand sides ``F`` of shape ``(P, m)`` or ``(P, m, r)``.

        ``bc`` optionally supplies Dirichlet data on the root boundary (open
        surfaces).  Returns nodal values of shape ``(P, N)`` or ``(P, N, r)``.
        """
        if self._compiled is None:
            raise RuntimeError("call factor() before solve()")
        F = np.asarray(F)
        squeeze = F.ndim == 2
        if squeeze:
            F = F[:, :, None]
        compiled = self._compiled
        if bc is not None:
            bc = np.asarray(bc).reshape(compiled.root_b, -1)
            if bc.shape[1] == 1 and F.shape[2] > 1:
                bc = np.repeat(bc, F.shape[2], axis=1)

        if np.issubdtype(compiled.dtype, np.complexfloating):
            U = compiled.solve(np.ascontiguousarray(F, dtype=np.result_type(F, compiled.dtype)), bc)
        else:
            Fr, was_complex = _as_real_view(F)
            if was_complex and bc is not None:
                bc = np.ascontiguousarray(bc.astype(np.complex128)).view(np.float64)
            U = compiled.solve(np.ascontiguousarray(Fr, dtype=np.float64), bc)
            if was_complex:
                U = np.ascontiguousarray(U).view(np.complex128)
        return U[:, :, 0] if squeeze else U

    def stats(self) -> dict[str, float]:
        """Summary of the factored tree: separator sizes, memory, and flop counts."""
        if self.root is None:
            raise RuntimeError("call factor() first")
        ks: list[int] = []
        bs: list[int] = []
        stack: list[Patch] = [self.root]
        while stack:
            node = stack.pop()
            if isinstance(node, Parent):
                ks.append(node.nseparator)
                bs.append(node.nboundary + 0)
                stack.extend((node.child1, node.child2))  # type: ignore[arg-type]
        k = np.asarray(ks, dtype=float)
        b = np.asarray(bs, dtype=float)
        compiled = self._compiled
        assert compiled is not None
        return {
            "npatches": self.leaf_ops.npatches,
            "nmerges": len(ks),
            "max_separator": int(k.max()) if k.size else 0,
            "empty_merges": int(np.sum(k == 0)),
            "levels": len(compiled.levels),
            "groups": sum(len(level) for level in compiled.levels),
            "factor_gflop": float(np.sum(2 * k**3 + 2 * k**2 * b + 2 * b**2 * k)) / 1e9,
            "solve_mflop_per_rhs": float(np.sum(2 * k * k + 4 * k * b)) / 1e6,
        }
