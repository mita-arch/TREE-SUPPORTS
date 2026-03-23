"""
PSO Optimization  (v6 — correct 45° enforcement after z-move)
==============================================================

FIX: _enforce_angle_constraint() was detection-only (did `pass`).

New approach in _decode():
  After setting each node's z from the PSO vector, call
  tree.valid_z_window(node_id) which computes the exact [z_lo, z_hi]
  that keeps every branch to/from that node within 45°, given the
  CURRENT positions of its parent and children.  Then clamp z to
  that window.

  This must be done in topological order (leaves → roots) so that
  when we clamp a node's z, its children's z values are already final.

Why XY is not adjusted by PSO
  The paper optimises z and diameter only.  XY is fixed by topology.
  Changing XY would require re-running collision checks.
  The 45° constraint in XY is guaranteed by construction in topology.py
  (all lean steps checked before placement).
"""

import numpy as np
from scipy.spatial import KDTree
from .topology import Tree, MAX_BRANCH_ANGLE_DEG

W_START = 0.9
W_END   = 0.4
C1      = 2.0
C2      = 2.0
V_MAX_Z    = 15.0
V_MAX_DIAM = 0.5
GREEDY_SEARCH_RADIUS = 30.0


def _topological_order(tree: Tree) -> list:
    """Return node IDs leaves-first, roots-last (correct propagation order)."""
    in_deg = {nid: 0 for nid in tree.nodes}
    for node in tree.nodes.values():
        if node.parent_id >= 0:
            in_deg[node.parent_id] = in_deg.get(node.parent_id, 0) + 1

    queue  = [nid for nid, d in in_deg.items() if d == 0]
    order  = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        node = tree.nodes[nid]
        if node.parent_id >= 0:
            in_deg[node.parent_id] -= 1
            if in_deg[node.parent_id] == 0:
                queue.append(node.parent_id)

    for nid in tree.nodes:
        if nid not in order:
            order.append(nid)
    return order


def _encode(tree: Tree, branch_ids: list) -> np.ndarray:
    x = np.zeros(len(branch_ids) * 2, dtype=float)
    for i, nid in enumerate(branch_ids):
        n = tree.nodes[nid]
        x[2*i]   = n.position[2]
        x[2*i+1] = n.diameter
    return x


def _decode(x: np.ndarray, tree_template: Tree, branch_ids: list,
            min_diam: float, max_diam: float) -> Tree:
    """
    Apply PSO vector to a cloned tree, then re-enforce:
      1. Diameter constraint  (parent >= child, propagate leaf→root)
      2. 45° angle constraint (clamp z using valid_z_window, leaf→root)
    """
    t = tree_template.clone()

    # ── Apply raw PSO values ──────────────────────────────────────────────────
    for i, nid in enumerate(branch_ids):
        node = t.nodes[nid]
        node.position[2] = float(np.clip(x[2*i],   node.z_min, node.z_max))
        node.diameter    = float(np.clip(x[2*i+1], min_diam,   max_diam))

    # ── Propagate diameter constraint (leaves → roots) ────────────────────────
    order = _topological_order(t)
    for nid in order:
        node = t.nodes[nid]
        node.diameter = float(np.clip(node.diameter, min_diam, max_diam))
        if node.parent_id >= 0:
            parent = t.nodes[node.parent_id]
            if parent.diameter < node.diameter:
                parent.diameter = node.diameter

    # ── Enforce 45° angle by clamping z (leaves → roots) ─────────────────────
    # Process in topological order so children are finalised before parents.
    branch_set = set(branch_ids)
    for nid in order:
        if nid not in branch_set:
            continue   # leaves and roots have fixed z
        node = t.nodes[nid]
        z_lo, z_hi = t.valid_z_window(nid)
        node.position[2] = float(np.clip(node.position[2], z_lo, z_hi))

    return t


def _cylinder_volume(tree: Tree) -> float:
    total = 0.0
    for node in tree.nodes.values():
        if node.parent_id < 0:
            continue
        parent = tree.nodes[node.parent_id]
        l = np.linalg.norm(node.position - parent.position)
        total += (np.pi / 4.0) * node.diameter**2 * l
    return total


def greedy_surface_attach(tree: Tree, kdtree: KDTree, verts: np.ndarray):
    improved = 0
    for nid in list(tree.nodes.keys()):
        node = tree.nodes.get(nid)
        if node is None or node.node_type != 'branch' or node.parent_id < 0:
            continue
        parent = tree.nodes[node.parent_id]
        cur_len = np.linalg.norm(node.position - parent.position)

        idxs = kdtree.query_ball_point(node.position, r=GREEDY_SEARCH_RADIUS)
        if not idxs:
            continue
        cands = verts[idxs]
        below = cands[cands[:, 2] < node.position[2] - 1.0]
        if len(below) == 0:
            continue

        dists = np.linalg.norm(below - parent.position, axis=1)
        best  = np.argmin(dists)
        if dists[best] < cur_len:
            parent.position  = below[best].copy()
            parent.node_type = 'root'
            improved += 1



def optimize_tree(tree: Tree, mesh,
                  n_particles: int  = 20,
                  n_iterations: int = 200,
                  min_diameter: float = 0.5,
                  max_diameter: float = 2.5,
                  verbose_every: int  = 100):

    kdtree = KDTree(mesh.vertices)
    verts  = mesh.vertices

    branch_ids = [n.id for n in tree.branch_nodes()]
    D = len(branch_ids)

    print(f"    [pso] {D} branch nodes | "
          f"{n_particles}p × {n_iterations}it")

    if D == 0:
        return tree.clone(), _cylinder_volume(tree)

    X0 = _encode(tree, branch_ids)
    X  = np.tile(X0, (n_particles, 1)).astype(float)

    # Small perturbations
    perturb = np.random.normal(0, 2.0, X.shape)
    perturb[:, 1::2] *= 0.2
    X += perturb
    X[:, ::2]  = np.clip(X[:, ::2],  0.0, None)
    X[:, 1::2] = np.clip(X[:, 1::2], min_diameter, max_diameter)

    V = np.zeros_like(X)

    print(f"    [pso] Initialising swarm...")
    pBest        = X.copy()
    pBest_scores = np.full(n_particles, np.inf)
    pBest_trees  = [None] * n_particles

    for i in range(n_particles):
        t = _decode(X[i], tree, branch_ids, min_diameter, max_diameter)
        greedy_surface_attach(t, kdtree, verts)
        s = _cylinder_volume(t)
        pBest_scores[i] = s
        pBest_trees[i]  = t

    gi          = int(np.argmin(pBest_scores))
    gBest       = pBest[gi].copy()
    gBest_score = pBest_scores[gi]
    gBest_tree  = pBest_trees[gi]

    print(f"    [pso] Initial gBest={gBest_score:.2f} mm3")

    for k in range(n_iterations):
        w  = W_START - (W_START - W_END) * k / max(n_iterations - 1, 1)
        r1 = np.random.random((n_particles, D*2))
        r2 = np.random.random((n_particles, D*2))

        V = w*V + C1*r1*(pBest - X) + C2*r2*(gBest - X)
        V[:, ::2]  = np.clip(V[:, ::2],  -V_MAX_Z,    V_MAX_Z)
        V[:, 1::2] = np.clip(V[:, 1::2], -V_MAX_DIAM, V_MAX_DIAM)
        X = X + V
        X[:, ::2]  = np.clip(X[:, ::2],  0.0, None)
        X[:, 1::2] = np.clip(X[:, 1::2], min_diameter, max_diameter)

        for i in range(n_particles):
            t = _decode(X[i], tree, branch_ids, min_diameter, max_diameter)
            greedy_surface_attach(t, kdtree, verts)
            s = _cylinder_volume(t)

            if s < pBest_scores[i]:
                pBest[i]        = X[i].copy()
                pBest_scores[i] = s
                pBest_trees[i]  = t

                if s < gBest_score:
                    gBest       = X[i].copy()
                    gBest_score = s
                    gBest_tree  = t

        if (k+1) % verbose_every == 0 or k == n_iterations - 1:
            mean = np.mean(pBest_scores[np.isfinite(pBest_scores)])
            print(f"    [pso] iter {k+1:4d}: gBest={gBest_score:.2f}  "
                  f"mean={mean:.2f}  w={w:.3f}")

    print(f"    [pso] Done → gBest={gBest_score:.2f} mm3")
    return gBest_tree, gBest_score
