"""
Tree Topology Generation — Phase 2.2  (v3 — collision-aware + outward lean)
----------------------------------------------------------------------------

Changes from v2
---------------
  - Grid columns that fall inside or too close to the model are excluded
    BEFORE any topology is built. This stops branches spawning inside the stem.

  - Merge node positions are collision-checked against the model. If the
    candidate position or the path to it clips the mesh, it is skipped.

  - Descent adds an OUTWARD LEAN: each step moves the XY slightly away
    from the model's centroid, so branches fan outward like real tree supports
    instead of falling straight down as poles.

  - Straight-vertical pole guard: if a descent segment would be closer than
    MIN_LEAN_FRAC of the grid resolution to perfectly vertical, a minimum
    horizontal offset is forced.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

MAX_BRANCH_ANGLE_DEG = 30.0
MAX_CHILDREN_PER_NODE = 6
MERGE_DIAM_GROWTH     = 1.20
DESCENT_DIAM_GROWTH   = 1.08

# Outward lean per descent level, as a fraction of grid XY resolution.
# 0.15 = move 15% of grid_resolution outward from model centre each level.
LEAN_FRAC    = 0.18
MIN_LEAN_MM  = 1.2   # minimum absolute lean per level (mm)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    id: int
    position: np.ndarray
    diameter: float
    node_type: str
    parent_id: int = -1
    children_ids: List[int] = field(default_factory=list)
    col_idx: int = -1
    level_idx: int = -1
    z_min: float = 0.0
    z_max: float = float('inf')

    def copy(self):
        return TreeNode(
            id=self.id, position=self.position.copy(),
            diameter=self.diameter, node_type=self.node_type,
            parent_id=self.parent_id, children_ids=list(self.children_ids),
            col_idx=self.col_idx, level_idx=self.level_idx,
            z_min=self.z_min, z_max=self.z_max,
        )


class Tree:
    def __init__(self):
        self.nodes: Dict[int, TreeNode] = {}
        self._next_id = 0

    def add_node(self, position, diameter, node_type,
                 col_idx=-1, level_idx=-1, z_min=0.0, z_max=None):
        pos = np.asarray(position, dtype=float)
        if z_max is None:
            z_max = pos[2] + 1000.0
        node = TreeNode(
            id=self._next_id, position=pos,
            diameter=float(diameter), node_type=node_type,
            col_idx=col_idx, level_idx=level_idx,
            z_min=z_min, z_max=z_max,
        )
        self.nodes[self._next_id] = node
        self._next_id += 1
        return node.id

    def connect(self, child_id: int, parent_id: int):
        child  = self.nodes[child_id]
        parent = self.nodes[parent_id]
        child.parent_id = parent_id
        if child_id not in parent.children_ids:
            parent.children_ids.append(child_id)

    def branch_nodes(self): return [n for n in self.nodes.values() if n.node_type == 'branch']
    def leaf_nodes(self):   return [n for n in self.nodes.values() if n.node_type == 'leaf']
    def root_nodes(self):   return [n for n in self.nodes.values() if n.node_type == 'root']

    def compute_volume(self) -> float:
        total = 0.0
        for node in self.nodes.values():
            if node.parent_id < 0:
                continue
            parent = self.nodes[node.parent_id]
            h  = np.linalg.norm(node.position - parent.position)
            r1 = node.diameter / 2.0
            r2 = parent.diameter / 2.0
            total += (np.pi / 3.0) * h * (r1**2 + r1*r2 + r2**2)
        return total

    def clone(self):
        t = Tree()
        t._next_id = self._next_id
        for nid, node in self.nodes.items():
            t.nodes[nid] = node.copy()
        return t

    def __repr__(self):
        return (f"Tree(nodes={len(self.nodes)}, "
                f"leaves={len(self.leaf_nodes())}, "
                f"branches={len(self.branch_nodes())}, "
                f"roots={len(self.root_nodes())}, "
                f"vol={self.compute_volume():.1f} mm3)")


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _angle_ok(p_child: np.ndarray, p_parent: np.ndarray) -> bool:
    delta = p_child - p_parent
    dz    = delta[2]
    dxy   = np.linalg.norm(delta[:2])
    if dz < 1e-9:
        return False
    return np.degrees(np.arctan2(dxy, dz)) >= MAX_BRANCH_ANGLE_DEG


def _outward_direction(xy: np.ndarray, model_center_xy: np.ndarray) -> np.ndarray:
    """
    Return a unit 2D vector pointing away from the model centre.
    Falls back to a random direction if the point IS the centre.
    """
    diff = xy - model_center_xy
    norm = np.linalg.norm(diff)
    if norm < 1e-6:
        angle = np.random.uniform(0, 2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)])
    return diff / norm


# ─── Hierarchical topology builder ────────────────────────────────────────────

def build_hierarchical_topology(support_points: np.ndarray,
                                 grid,
                                 safe_cols: set = None,
                                 mesh=None,
                                 model_center_xy: np.ndarray = None,
                                 tip_diameter: float = 0.5,
                                 max_diameter: float = 3.0,
                                 n_merge_passes: int = 4,
                                 seed: int = None) -> 'Tree':
    """
    Build a collision-aware hierarchical tree support.

    Parameters
    ----------
    support_points   : (N,3) overhang attachment positions
    grid             : Grid
    safe_cols        : set of column indices clear of the model (None = all)
    mesh             : trimesh.Trimesh, used for segment collision checks
    model_center_xy  : np.ndarray (2,), XY centroid of the model for lean direction
    tip_diameter     : mm at leaf nodes
    max_diameter     : cap on trunk diameter
    n_merge_passes   : descend+merge cycles
    seed             : random seed
    """
    if seed is not None:
        np.random.seed(seed)

    if safe_cols is None:
        safe_cols = set(range(grid.n_cols))

    if model_center_xy is None:
        model_center_xy = np.array([
            (grid.bounds[0][0] + grid.bounds[1][0]) / 2.0,
            (grid.bounds[0][1] + grid.bounds[1][1]) / 2.0,
        ])

    # Lazy import collision checker
    if mesh is not None:
        from src.collision import segment_clears_mesh, point_clears_mesh
        def seg_ok(a, b): return segment_clears_mesh(a, b, mesh)
        def pt_ok(p):     return point_clears_mesh(p, mesh)
    else:
        def seg_ok(a, b): return True
        def pt_ok(p):     return True

    tree = Tree()
    if len(support_points) == 0:
        return tree

    lean_step = max(MIN_LEAN_MM, grid.xy_resolution * LEAN_FRAC)

    # ── STEP 1: Assign leaves to safe columns only ────────────────────────────

    col_to_leaves: Dict[int, List[int]] = {}
    skipped_unsafe = 0

    for pt in support_points:
        # Find nearest SAFE column
        xy   = pt[:2]
        dists = np.linalg.norm(grid.col_positions - xy, axis=1)

        # Try up to 8 nearest columns, pick first that is safe
        nearest = np.argsort(dists)
        chosen  = None
        for idx in nearest[:8]:
            if int(idx) in safe_cols:
                chosen = int(idx)
                break

        if chosen is None:
            skipped_unsafe += 1
            continue

        lid = tree.add_node(
            position=pt, diameter=tip_diameter,
            node_type='leaf', col_idx=chosen,
            z_min=pt[2], z_max=pt[2],
        )
        col_to_leaves.setdefault(chosen, []).append(lid)

    if skipped_unsafe:
        print(f"    [topo] Skipped {skipped_unsafe} leaves with no safe nearby column")

    if not col_to_leaves:
        print("    [topo] WARNING: no leaves placed — all columns blocked")
        return tree

    print(f"    [topo] {len(support_points)} leaves on "
          f"{len(col_to_leaves)} safe columns ({skipped_unsafe} skipped)")

    # ── STEP 2: Per-column gather — merge leaves into first branch node ────────

    active: Dict[int, Tuple[int, int, np.ndarray]] = {}
    # active maps col_idx → (node_id, level_idx, current_xy)

    for col_idx, leaf_ids in col_to_leaves.items():
        min_z       = min(tree.nodes[lid].position[2] for lid in leaf_ids)
        start_level = grid.level_below(min_z)

        n           = len(leaf_ids)
        branch_diam = min(tip_diameter * (n ** 0.5) * MERGE_DIAM_GROWTH, max_diameter)
        branch_pos  = grid.get_node_position(col_idx, start_level)
        branch_type = 'root' if start_level == 0 else 'branch'

        bid = tree.add_node(
            position=branch_pos, diameter=branch_diam,
            node_type=branch_type, col_idx=col_idx,
            level_idx=start_level,
            z_min=grid.z_floor, z_max=branch_pos[2],
        )
        for lid in leaf_ids:
            tree.connect(lid, bid)

        active[col_idx] = (bid, start_level, branch_pos[:2].copy())

    print(f"    [topo] After column gather: {len(active)} active tips")

    # ── STEP 3: Iterative descend + merge ─────────────────────────────────────

    n_levels       = grid.n_levels
    descent_levels = max(1, (n_levels - 1) // max(n_merge_passes + 1, 2))
  

    for pass_i in range(n_merge_passes):

               # ── 3a. Descend each active tip with outward lean ─────────────────────
        new_active: Dict[int, Tuple[int, int, np.ndarray]] = {}

        for col_idx, (tip_id, tip_level, cur_xy) in active.items():
            tip_node  = tree.nodes[tip_id]
            target_lv = max(0, tip_level - descent_levels)

            if target_lv == tip_level:
                new_active[col_idx] = (tip_id, tip_level, cur_xy)
                continue

            new_diam = min(tip_node.diameter * (DESCENT_DIAM_GROWTH ** descent_levels),
                           max_diameter)

            # Compute outward lean
            out_dir  = _outward_direction(cur_xy, model_center_xy)
            dz       = grid.levels[tip_level] - grid.levels[target_lv]

            # Disable leaning near the ground
            if target_lv <= 1:
                lean_xy = cur_xy.copy()
            else:
                lean_xy = cur_xy + out_dir * lean_step * descent_levels

            # Clamp lean within grid bounds with 4mm margin
            x_lo, x_hi = grid.bounds[0][0] - 4, grid.bounds[1][0] + 4
            y_lo, y_hi = grid.bounds[0][1] - 4, grid.bounds[1][1] + 4
            lean_xy[0] = np.clip(lean_xy[0], x_lo, x_hi)
            lean_xy[1] = np.clip(lean_xy[1], y_lo, y_hi)

            new_z    = grid.levels[target_lv]
            new_pos  = np.array([lean_xy[0], lean_xy[1], new_z])
            new_type = 'root' if target_lv == 0 else 'branch'

            # 45° angle check
            if not _angle_ok(tip_node.position, new_pos):
                # Reduce lean until angle is OK
                for frac in [0.5, 0.25, 0.0]:
                    if target_lv <= 1:
                        clamped_xy = cur_xy.copy()
                    else:
                        clamped_xy = cur_xy + out_dir * lean_step * descent_levels * frac
                    new_pos    = np.array([clamped_xy[0], clamped_xy[1], new_z])
                    if _angle_ok(tip_node.position, new_pos):
                        lean_xy = clamped_xy
                        break

            # Collision check against model
            if not seg_ok(tip_node.position, new_pos) or not pt_ok(new_pos):
                # Try with no lean (straight down)
                fallback = np.array([cur_xy[0], cur_xy[1], new_z])
                if seg_ok(tip_node.position, fallback) and pt_ok(fallback):
                    new_pos = fallback
                    lean_xy = cur_xy.copy()
                else:
                    # Can't descend cleanly — stay at current level
                    new_active[col_idx] = (tip_id, tip_level, cur_xy)
                    continue

            new_id = tree.add_node(
                position=new_pos, diameter=new_diam,
                node_type=new_type, col_idx=-1,
                level_idx=target_lv,
                z_min=grid.z_floor, z_max=new_pos[2],
            )
            tree.connect(tip_id, new_id)
            new_active[col_idx] = (new_id, target_lv, lean_xy)

        active = new_active

        # ── 3b. Merge neighbouring active tips ────────────────────────────────
        col_list   = list(active.keys())
        np.random.shuffle(col_list)
        merged_set = set()
        n_merged   = 0

        for col_a in col_list:
            if col_a in merged_set:
                continue
            tip_a_id, level_a, xy_a = active[col_a]
            node_a = tree.nodes[tip_a_id]

            if len(node_a.children_ids) >= MAX_CHILDREN_PER_NODE:
                continue

            neighbors = grid.get_neighbor_columns(col_a, max_neighbors=6)

            for col_b in neighbors:
                if col_b not in active or col_b in merged_set or col_b == col_a:
                    continue
                tip_b_id, level_b, xy_b = active[col_b]
                node_b = tree.nodes[tip_b_id]

                merge_level = max(0, min(level_a, level_b) - 2)
                merge_z     = grid.levels[merge_level]

                # Merge XY: blend, biased slightly outward from model centre
                alpha     = 0.1 + 0.1 * np.random.random()
                #alpha     = 0.5
                blend_xy = alpha * (xy_a + xy_b)
                out_dir   = _outward_direction(blend_xy, model_center_xy)
                merge_xy  = blend_xy + out_dir * (lean_step * 0.05)

                # Clamp within extended bounds
                merge_xy[0] = np.clip(merge_xy[0],
                                      grid.bounds[0][0] - 4, grid.bounds[1][0] + 4)
                merge_xy[1] = np.clip(merge_xy[1],
                                      grid.bounds[0][1] - 4, grid.bounds[1][1] + 4)

                merge_pos = np.array([merge_xy[0], merge_xy[1], merge_z])

                # Angle check
                if not _angle_ok(node_a.position, merge_pos):
                    continue
                if not _angle_ok(node_b.position, merge_pos):
                    continue

                # Collision check
                if not pt_ok(merge_pos):
                    continue
                if not seg_ok(node_a.position, merge_pos):
                    continue
                if not seg_ok(node_b.position, merge_pos):
                    continue

                merge_diam = min(
                    np.sqrt(node_a.diameter**2 + node_b.diameter**2) * MERGE_DIAM_GROWTH,
                    max_diameter
                )
                merge_type = 'root' if merge_level == 0 else 'branch'

                mid = tree.add_node(
                    position=merge_pos, diameter=merge_diam,
                    node_type=merge_type, col_idx=-1,
                    level_idx=merge_level,
                    z_min=grid.z_floor, z_max=merge_pos[2],
                )
                tree.connect(tip_a_id, mid)
                tree.connect(tip_b_id, mid)

                active[col_a] = (mid, merge_level, merge_xy)
                del active[col_b]
                merged_set.add(col_b)
                n_merged += 1
                break

        if len(active) <= 1:
            break

    # ── STEP 4: Ground remaining tips ─────────────────────────────────────────
    grounded = 0
    for col_idx, (tip_id, tip_level, cur_xy) in active.items():
        node = tree.nodes[tip_id]
        if tip_level == 0:
            node.node_type = 'root'
            continue

        cur_id   = tip_id
        cur_diam = node.diameter
        xy       = cur_xy.copy()

        for lv in range(tip_level - 1, -1, -1):
            cur_node = tree.nodes[cur_id]
            out_dir  = _outward_direction(xy, model_center_xy)
            xy       = xy + out_dir * lean_step

            # Clamp
            xy[0] = np.clip(xy[0], grid.bounds[0][0] - 5, grid.bounds[1][0] + 5)
            xy[1] = np.clip(xy[1], grid.bounds[0][1] - 5, grid.bounds[1][1] + 5)

            new_z    = grid.levels[lv]
            new_pos  = np.array([xy[0], xy[1], new_z])

            # Angle check — reduce lean if too steep
            if not _angle_ok(cur_node.position, new_pos):
                for frac in [0.5, 0.25, 0.0]:
                    xy_try   = cur_xy + out_dir * lean_step * frac
                    new_pos  = np.array([xy_try[0], xy_try[1], new_z])
                    if _angle_ok(cur_node.position, new_pos):
                        xy = xy_try
                        break

            # Collision check
            if not pt_ok(new_pos) or not seg_ok(cur_node.position, new_pos):
                # Fall back: no lean
                new_pos = np.array([cur_node.position[0], cur_node.position[1], new_z])
                xy      = new_pos[:2].copy()

            cur_diam = min(cur_diam * DESCENT_DIAM_GROWTH, max_diameter)
            ntype    = 'root' if lv == 0 else 'branch'
            nid      = tree.add_node(
                position=new_pos, diameter=cur_diam,
                node_type=ntype, col_idx=-1,
                level_idx=lv,
                z_min=grid.z_floor, z_max=new_pos[2],
            )
            tree.connect(cur_id, nid)
            cur_id = nid

        grounded += 1

    if grounded:
        print(f"    [topo] Grounded {grounded} remaining tip(s) with outward lean")


    print(f"    [topo] Final: {tree}")
    return tree


# ─── Pool generator ───────────────────────────────────────────────────────────

def generate_topology_pool(support_points: np.ndarray,
                            grid,
                            mesh=None,
                            safe_cols: set = None,
                            model_center_xy: np.ndarray = None,
                            n_topologies: int = 2,
                            pool_multiplier: int = 5,
                            tip_diameter: float = 0.5,
                            max_diameter: float = 3.0,
                            n_merge_passes: int = 4) -> List['Tree']:
    """
    Generate hierarchical tree candidates (collision-aware) and return
    the best n_topologies ranked by volume.
    """
    n_pool = n_topologies * pool_multiplier
    print(f"  [topology] Generating {n_pool} collision-aware candidates "
          f"(tip={tip_diameter:.2f} mm, max={max_diameter:.2f} mm, "
          f"passes={n_merge_passes})...")

    pool = []

    for i in range(n_pool):
        try:
            tree = build_hierarchical_topology(
                support_points  = support_points,
                grid            = grid,
                safe_cols       = safe_cols,
                mesh            = mesh,
                model_center_xy = model_center_xy,
                tip_diameter    = tip_diameter,
                max_diameter    = max_diameter,
                n_merge_passes  = n_merge_passes,
                seed            = None,
            )
            vol = tree.compute_volume()
            if vol < 1e-3 or len(tree.leaf_nodes()) == 0:
                continue
            pool.append((vol, tree))

            if (i + 1) % max(1, n_pool // 4) == 0:
                best = min(v for v, _ in pool) if pool else float('inf')
                print(f"  [topology] {i+1}/{n_pool} | valid={len(pool)} | "
                      f"best_vol={best:.1f} mm3")

        except Exception as e:
            print(f"  [topology] WARN candidate {i}: {e}")
            import traceback; traceback.print_exc()

    if not pool:
        raise RuntimeError(
            "No valid topologies generated. "
            "Try: --grid_resolution smaller, --sample_spacing smaller, "
            "or --clearance smaller."
        )

    pool.sort(key=lambda x: x[0])
    best = pool[:n_topologies]

    print(f"\n  [topology] {len(pool)} valid, returning best {len(best)}:")
    for rank, (vol, t) in enumerate(best):
        print(f"    Rank {rank+1}: leaves={len(t.leaf_nodes())} "
              f"branches={len(t.branch_nodes())} "
              f"roots={len(t.root_nodes())} vol={vol:.1f} mm3")

    return [t for _, t in best]
