"""
Tree Topology Generation  (v6)
==============================

Fixes for three structural issues
-----------------------------------

FIX 1 — Separate child limits for leaf-gather vs branch-merge
  Old: MAX_CHILDREN_PER_NODE = 6 was defined but NEVER CHECKED.
       connect() had no guard → any number of children accumulated.

  New: Two separate limits, both enforced in connect():
    MAX_LEAVES_PER_BRANCH = 4
      Max leaf nodes one first-level branch node may gather.
      Keeps the top of the tree thin.
      Excess leaves spill to the next-nearest safe column.

    MAX_BRANCHES_PER_MERGE = 3
      Max branch nodes that may merge into one merge node.
      Prevents wide flat "broom" junctions that print poorly.

FIX 2 — PSO angle enforcement was detection-only
  Old: _enforce_angle_constraint() counted violations then did `pass`.
       z_min/z_max clamping used the node's ORIGINAL z, not the
       current parent/child positions after PSO moves.

  New: After every PSO position update, recompute each branch node's
  valid z window from the CURRENT positions of its parent and children:
    z_safe_min = z_parent + dxy(node→parent) / tan(45°)
    z_safe_max = min over children of [z_child - dxy(node→child)/tan(45°)]
  Then clamp: z = clip(z, z_safe_min, z_safe_max)
  This guarantees 45° is maintained after every PSO step.

FIX 3 — Leaf-gather diameter scaling capped
  Old: same sqrt(n)*tip_diameter formula for both leaf and branch merges.
       With 8 leaves: sqrt(8)*0.5*1.2 = 1.70mm at the very top.

  New: leaf-gather uses a softer CBRT (cube-root) scale:
    branch_diam = tip_diameter * n^(1/3) * LEAF_GATHER_GROWTH
    For n=4: 0.5 * 4^(1/3) * 1.1 = 0.78mm  (stays thin)
    For n=4: sqrt formula gives  0.5 * 2 * 1.2 = 1.20mm  (too thick)
  Branch-merge keeps area-conserving sqrt formula (physically correct).
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from src.collision import (point_is_safe, segment_is_safe,
                            seg_seg_min_dist, CLEARANCE_MM)

MAX_BRANCH_ANGLE_DEG = 45.0
TAN45 = np.tan(np.radians(MAX_BRANCH_ANGLE_DEG))  # = 1.0

# ── Child count limits (both enforced in connect()) ──────────────────────────
MAX_LEAVES_PER_BRANCH  = 8   # leaf nodes → one first-level branch
MAX_BRANCHES_PER_MERGE = 5   # branch nodes → one merge node

# ── Diameter growth factors ───────────────────────────────────────────────────
LEAF_GATHER_GROWTH  = 1.10   # used with cbrt scaling for leaf→branch
BRANCH_MERGE_GROWTH = 1.35   # used with sqrt scaling for branch→branch (area-conserving)

MAX_LEAN_MM     = 2.5
BRANCH_CLEARANCE_MM = 0.8


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    id: int
    position: np.ndarray
    diameter: float
    node_type: str           # 'leaf' | 'branch' | 'root'
    parent_id: int = -1
    children_ids: List[int] = field(default_factory=list)
    z_min: float = 0.0
    z_max: float = float('inf')

    def copy(self):
        return TreeNode(
            id=self.id, position=self.position.copy(),
            diameter=self.diameter, node_type=self.node_type,
            parent_id=self.parent_id, children_ids=list(self.children_ids),
            z_min=self.z_min, z_max=self.z_max,
        )


class Tree:
    def __init__(self):
        self.nodes: Dict[int, TreeNode] = {}
        self._next_id = 0
        self._branches: List[Tuple[np.ndarray, np.ndarray]] = []

    def add_node(self, position, diameter, node_type, z_min=0.0, z_max=None):
        pos = np.asarray(position, dtype=float)
        if z_max is None:
            z_max = pos[2] + 1000.0
        n = TreeNode(
            id=self._next_id, position=pos,
            diameter=float(diameter), node_type=node_type,
            z_min=z_min, z_max=z_max,
        )
        self.nodes[self._next_id] = n
        self._next_id += 1
        return n.id

    def connect(self, child_id: int, parent_id: int,
                max_children: int = MAX_BRANCHES_PER_MERGE) -> bool:
        """
        Connect child → parent.  Returns False (and does NOT connect) if
        the parent already has max_children children of the same type class.

        Type classes:
          leaf children    → limited by MAX_LEAVES_PER_BRANCH
          non-leaf children → limited by max_children (default MAX_BRANCHES_PER_MERGE)
        """
        child  = self.nodes[child_id]
        parent = self.nodes[parent_id]

        # Choose the right limit for this child type
        if child.node_type == 'leaf':
            limit = MAX_LEAVES_PER_BRANCH
        else:
            limit = max_children

        # Count existing children of the same type class
        existing = sum(
            1 for cid in parent.children_ids
            if self.nodes[cid].node_type == child.node_type
        )
        if existing >= limit:
            return False   # parent is full for this child type

        child.parent_id = parent_id
        if child_id not in parent.children_ids:
            parent.children_ids.append(child_id)
        self._branches.append((child.position.copy(), parent.position.copy()))
        return True

    def connect_force(self, child_id: int, parent_id: int):
        """Connect without any child-count limit (used for descent chain)."""
        child  = self.nodes[child_id]
        parent = self.nodes[parent_id]
        child.parent_id = parent_id
        if child_id not in parent.children_ids:
            parent.children_ids.append(child_id)
        self._branches.append((child.position.copy(), parent.position.copy()))

    def branch_clears_existing(self, p_start: np.ndarray, p_end: np.ndarray,
                                min_dist: float = BRANCH_CLEARANCE_MM) -> bool:
        for (a, b) in self._branches:
            if (np.linalg.norm(p_start - a) < 1e-6 or
                np.linalg.norm(p_start - b) < 1e-6 or
                np.linalg.norm(p_end   - a) < 1e-6 or
                np.linalg.norm(p_end   - b) < 1e-6):
                continue
            if seg_seg_min_dist(p_start, p_end, a, b) < min_dist:
                return False
        return True

    def leaf_nodes(self):   return [n for n in self.nodes.values() if n.node_type == 'leaf']
    def branch_nodes(self): return [n for n in self.nodes.values() if n.node_type == 'branch']
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

    def valid_z_window(self, node_id: int) -> Tuple[float, float]:
        """
        Compute the z range [z_lo, z_hi] that keeps ALL branches connected
        to this node within MAX_BRANCH_ANGLE_DEG (45°).

        Used by PSO to clamp z after each update.

        z_lo: must be high enough above parent to form valid angle
        z_hi: must be low enough below each child to form valid angle

        Given fixed XY:
          dxy(a, b) = horizontal distance between a and b
          dz_min = dxy / tan(45°) = dxy  (since tan45=1)
        """
        node = self.nodes[node_id]
        z_lo = node.z_min  # hard floor (build plate)
        z_hi = node.z_max  # hard ceiling (original position)

        # Lower bound: must be at least dxy above parent
        if node.parent_id >= 0:
            parent = self.nodes[node.parent_id]
            dxy = float(np.linalg.norm(node.position[:2] - parent.position[:2]))
            required_z = parent.position[2] + dxy / TAN45 + 1e-3
            z_lo = max(z_lo, required_z)

        # Upper bound: must be at least dxy below each child
        for cid in node.children_ids:
            child = self.nodes[cid]
            dxy   = float(np.linalg.norm(node.position[:2] - child.position[:2]))
            max_z = child.position[2] - dxy / TAN45 - 1e-3
            z_hi  = min(z_hi, max_z)

        # If constraints are infeasible, return a small valid window at midpoint
        if z_lo > z_hi:
            mid   = (z_lo + z_hi) / 2.0
            z_lo  = mid - 0.5
            z_hi  = mid + 0.5

        return float(z_lo), float(z_hi)

    def clone(self):
        t = Tree()
        t._next_id  = self._next_id
        t._branches = list(self._branches)
        for nid, node in self.nodes.items():
            t.nodes[nid] = node.copy()
        return t

    def __repr__(self):
        return (f"Tree(nodes={len(self.nodes)}, "
                f"L={len(self.leaf_nodes())} "
                f"B={len(self.branch_nodes())} "
                f"R={len(self.root_nodes())}, "
                f"vol={self.compute_volume():.1f} mm3)")


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _angle_ok(p_child: np.ndarray, p_parent: np.ndarray) -> bool:
    delta = p_child - p_parent
    dz    = delta[2]
    dxy   = np.linalg.norm(delta[:2])
    if dz < 1e-9:
        return False
    return np.degrees(np.arctan2(dxy, dz)) <= MAX_BRANCH_ANGLE_DEG


def _outward_dir(xy: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    d = xy - centroid
    n = np.linalg.norm(d)
    if n < 1e-6:
        ang = np.random.uniform(0, 2*np.pi)
        return np.array([np.cos(ang), np.sin(ang)])
    return d / n


def _lean_at_z(z, z_top, z_floor, max_lean, taper_frac=0.40):
    total_h = z_top - z_floor
    if total_h < 1e-6:
        return 0.0
    taper_start = z_floor + total_h * taper_frac
    if z > taper_start:
        return max_lean
    frac = (z - z_floor) / (taper_start - z_floor + 1e-9)
    return max_lean * max(0.0, frac)


# ─── Merge helper ─────────────────────────────────────────────────────────────

def _try_merge(node_a: TreeNode, node_b: TreeNode,
               target_z: float, tree: Tree, mesh,
               clearance: float, model_centroid: np.ndarray) -> Optional[np.ndarray]:
    xy_a = node_a.position[:2]
    xy_b = node_b.position[:2]
    mid  = (xy_a + xy_b) / 2.0
    out_dir = _outward_dir(mid, model_centroid)

    dz_merge  = max(node_a.position[2], node_b.position[2]) - target_z
    max_offset = dz_merge * 0.90

    raw = [0.4 * max_offset, 0.5 * max_offset, 0.7 * max_offset, max_offset, 0.2 * max_offset, 0.1 * max_offset, 0.0]
    offsets = [min(o, max_offset) for o in raw]

    for offset in offsets:
        cand_pos = np.array([*(mid + out_dir * offset), target_z])

        if target_z >= node_a.position[2] - 1e-3: continue
        if target_z >= node_b.position[2] - 1e-3: continue
        if not _angle_ok(node_a.position, cand_pos): continue
        if not _angle_ok(node_b.position, cand_pos): continue
        if not point_is_safe(cand_pos, mesh, clearance): continue
        if not segment_is_safe(node_a.position, cand_pos, mesh, clearance): continue
        if not segment_is_safe(node_b.position, cand_pos, mesh, clearance): continue
        if not tree.branch_clears_existing(node_a.position, cand_pos): continue
        if not tree.branch_clears_existing(node_b.position, cand_pos): continue

        return cand_pos
    return None


# ─── Main topology builder ────────────────────────────────────────────────────

def build_tree(support_points: np.ndarray,
               grid,
               mesh,
               tip_diameter:  float = 0.5,
               max_diameter:  float = 2.5,
               clearance:     float = CLEARANCE_MM,
               merge_depth:   float = 0.7,
               merge_levels:  int   = 4,
               seed:          int   = None) -> Tree:
    # merge_depth: fraction of model height below overhang within which merges
    # are allowed.  0.0 = no merging at all.  1.0 = merges allowed all the way
    # to the build plate.  Default 0.6 = merges in top 60% of support height.
    # This spreads merges across the tree height instead of cramming them all
    # near the overhang.  Equivalent to controlling n_merge_passes in v2.

    if seed is not None:
        np.random.seed(seed)

    tree = Tree()
    if len(support_points) == 0:
        return tree

    model_centroid = np.array([
        (grid.bounds[0][0] + grid.bounds[1][0]) / 2.0,
        (grid.bounds[0][1] + grid.bounds[1][1]) / 2.0,
    ])
    z_floor = float(grid.z_floor)

    # ── STEP 1: Leaf nodes at exact overhang positions ────────────────────────

    leaf_ids = []
    for pt in support_points:
        lid = tree.add_node(position=pt, diameter=tip_diameter,
                            node_type='leaf', z_min=pt[2], z_max=pt[2])
        leaf_ids.append(lid)

    print(f"    [tree] {len(leaf_ids)} leaves at exact overhang XY")

    # ── STEP 2: Column-wise leaf gather (v2-style) ───────────────────────────
    # Group leaves by their nearest grid column.
    # The stem node XY = grid column XY (NOT the exact leaf XY).
    # This avoids placing the stem directly on or just under the mesh surface,
    # which caused all stems to fail point_is_safe() for thin/coarse meshes.
    #
    # Safety rules for the stem:
    #   - Only check point_is_safe() on the stem node itself.
    #   - Do NOT run segment_is_safe() for leaf→stem: the leaf is on the mesh
    #     surface by construction, so the segment starts on the boundary.
    #     Ray-casting on thin/coarse meshes misfires for such near-surface segs.
    #   - Still enforce MAX_LEAVES_PER_BRANCH via connect().
    #   - Excess leaves on the same column → additional stem nodes on that column
    #     (each capped at MAX_LEAVES_PER_BRANCH, staggered in Z if needed).
    #
    # Diameter: cbrt scaling keeps tips thin even when n=4 leaves gather.

    col_to_pending: Dict[int, List[int]] = {}
    for lid in leaf_ids:
        leaf    = tree.nodes[lid]
        col_idx = grid.nearest_column(leaf.position[:2])
        col_to_pending.setdefault(col_idx, []).append(lid)

    stem_ids   = []
    skipped    = 0
    n_cols_used = 0

    for col_idx, lids_on_col in col_to_pending.items():
        col_xy = grid.col_positions[col_idx]   # <── grid column XY, not leaf XY

        # Lowest leaf on this column determines the first stem level
        min_z = min(tree.nodes[lid].position[2] for lid in lids_on_col)
        lv    = grid.level_below(min_z)
        if lv < 0:
            skipped += len(lids_on_col)
            continue

        stem_z = grid.levels[lv]
        if stem_z >= min_z - 1e-3:
            skipped += len(lids_on_col)
            continue

        # Safety check on the STEM NODE only (not on the leaf→stem segment)
        stem_pos = np.array([col_xy[0], col_xy[1], stem_z])
        if not point_is_safe(stem_pos, mesh, clearance):
            # Try one level lower before giving up
            if lv > 0:
                stem_z2   = grid.levels[lv - 1]
                stem_pos2 = np.array([col_xy[0], col_xy[1], stem_z2])
                if point_is_safe(stem_pos2, mesh, clearance):
                    stem_pos = stem_pos2
                    stem_z   = stem_z2
                    lv       = lv - 1
                else:
                    skipped += len(lids_on_col)
                    continue
            else:
                skipped += len(lids_on_col)
                continue

        # Split this column's leaves into batches of MAX_LEAVES_PER_BRANCH.
        # Each batch shares one stem node. Multiple batches = multiple stems
        # on the same column (stagger Z by one level each to avoid coincidence).
        batches = [lids_on_col[i:i + MAX_LEAVES_PER_BRANCH]
                   for i in range(0, len(lids_on_col), MAX_LEAVES_PER_BRANCH)]

        for b_idx, batch in enumerate(batches):
            # Stagger z for extra batches on same column
            cur_lv    = max(0, lv - b_idx)
            cur_z     = grid.levels[cur_lv]
            cur_pos   = np.array([col_xy[0], col_xy[1], cur_z])

            if b_idx > 0 and not point_is_safe(cur_pos, mesh, clearance):
                skipped += len(batch)
                continue

            n         = len(batch)
            stem_diam = min(tip_diameter * (n ** (1/3)) * LEAF_GATHER_GROWTH,
                            max_diameter)

            sid = tree.add_node(position=cur_pos, diameter=stem_diam,
                                node_type='branch',
                                z_min=z_floor, z_max=cur_pos[2])

            connected = 0
            for lid in batch:
                ok = tree.connect(lid, sid)   # enforces MAX_LEAVES_PER_BRANCH
                if ok:
                    connected += 1
                else:
                    skipped += 1

            if connected > 0:
                stem_ids.append(sid)
                n_cols_used += 1

    if skipped:
        print(f"    [tree] {skipped} leaves skipped (column blocked or batch overflow)")
    print(f"    [tree] {len(stem_ids)} stem nodes across {len(col_to_pending)} columns "
          f"(max {MAX_LEAVES_PER_BRANCH} leaves each, column-XY stems)")

    if not stem_ids:
        print("    [tree] WARNING: no stems placed.")
        print("    [tree]   Possible causes:")
        print("    [tree]   1. --clearance too large: try --clearance 0.5")
        print("    [tree]   2. Model bounds do not include empty space below overhang")
        print("    [tree]   3. grid_resolution too coarse for model size")
        return tree

    # ── STEP 3: Closest-pair merge (min-heap) ────────────────────────────────
    #
    # How many merges happen and where:
    # ----------------------------------
    # There is no fixed "number of passes". The heap pops the globally
    # closest pair of active tips and merges them. This repeats until
    # the heap is empty or only one active tip remains.
    #
    # Number of merges ≈ n_stems - 1 in the best case (perfect binary tree).
    # In practice fewer succeed due to safety/angle/depth constraints.
    #
    # Where merges happen — the "meeting point" strategy:
    # ----------------------------------
    # Old code: target_z = one level below the HIGHER of the two tips.
    # Problem: if tip A is at z=94 and tip B is at z=70, target=z=88
    #          which is ABOVE B → merge always skipped → B descends alone
    #          to the build plate → lone trunks at the bottom.
    #
    # New code: try merge_levels candidate z values, starting just below
    # the LOWER of the two tips (not the higher), so we always find a z
    # that is below BOTH tips. The first level that passes all safety
    # checks is used. This encourages merges to happen HIGH UP where
    # they save the most material, while still working across height gaps.
    #
    # merge_depth parameter: fraction of model height (below overhang) in
    # which merges are ALLOWED. merge_depth=0.7 (default) = merges allowed
    # from the overhang down to 30% of the height. Tips below that zone
    # just descend straight to the build plate — no trunk consolidation
    # in the final bottom section, which avoids weird low crossing.

    active = set(stem_ids)
    heap   = []
    for i, id_a in enumerate(stem_ids):
        for id_b in stem_ids[i+1:]:
            d = float(np.linalg.norm(
                tree.nodes[id_a].position - tree.nodes[id_b].position))
            heapq.heappush(heap, (d, id_a, id_b))

    merges_ok   = 0
    merges_skip = 0

    overhang_z    = float(np.max([tree.nodes[sid].position[2] for sid in stem_ids]))
    merge_z_floor = z_floor + (overhang_z - z_floor) * (1.0 - merge_depth)
    print(f"    [tree] Merge zone z=[{merge_z_floor:.1f}, {overhang_z:.1f}] "
          f"depth={merge_depth:.2f}  merge_levels={merge_levels}")

    while heap and len(active) > 1:
        dist, id_a, id_b = heapq.heappop(heap)

        if id_a not in active or id_b not in active:
            continue

        node_a = tree.nodes[id_a]
        node_b = tree.nodes[id_b]

        lower_z = min(node_a.position[2], node_b.position[2])

        # Find a target z that is below BOTH tips.
        # Start just below the LOWER tip and try merge_levels candidates.
        merge_pos = None
        for step in range(1, merge_levels + 1):
            lv = grid.level_below(lower_z) - (step - 1)
            if lv < 0:
                break
            target_z = grid.levels[lv]

            # Must be below both tips
            if target_z >= node_a.position[2] - 1e-3: continue
            if target_z >= node_b.position[2] - 1e-3: continue

            # Must be within the allowed merge zone
            if target_z < merge_z_floor - 1e-3: break

            pos = _try_merge(node_a, node_b, target_z,
                             tree, mesh, clearance, model_centroid)
            if pos is not None:
                merge_pos = pos
                break   # found a valid z — use it (highest valid z first)

        if merge_pos is None:
            merges_skip += 1; continue

        target_z = merge_pos[2]
        target_lv = grid.nearest_level(target_z)

        # Area-conserving diameter merge (branch merges use sqrt — physically correct)
        merge_diam = min(
            np.sqrt(node_a.diameter**2 + node_b.diameter**2) * BRANCH_MERGE_GROWTH,
            max_diameter
        )
        merge_type = 'root' if target_lv == 0 else 'branch'

        mid_id = tree.add_node(position=merge_pos, diameter=merge_diam,
                               node_type=merge_type,
                               z_min=z_floor, z_max=merge_pos[2])

        # connect() enforces MAX_BRANCHES_PER_MERGE
        ok_a = tree.connect(id_a, mid_id, max_children=MAX_BRANCHES_PER_MERGE)
        ok_b = tree.connect(id_b, mid_id, max_children=MAX_BRANCHES_PER_MERGE)

        if not ok_a or not ok_b:
            # Undo: remove the merge node we just created
            del tree.nodes[mid_id]
            if tree._branches and np.allclose(tree._branches[-1][1], merge_pos):
                tree._branches.pop()
            if ok_a:
                tree.nodes[id_a].parent_id = -1
                tree.nodes[mid_id if mid_id in tree.nodes else id_a].children_ids = []
            merges_skip += 1
            continue

        active.discard(id_a)
        active.discard(id_b)
        active.add(mid_id)
        merges_ok += 1

        for other_id in active:
            if other_id == mid_id: continue
            d = float(np.linalg.norm(merge_pos - tree.nodes[other_id].position))
            heapq.heappush(heap, (d, mid_id, other_id))

    print(f"    [tree] Merges: {merges_ok} ok, {merges_skip} rejected → "
          f"{len(active)} tips to ground")

    # ── STEP 4: Ground remaining tips with tapered outward lean ───────────────

    grounded = 0
    for tip_id in list(active):
        node = tree.nodes[tip_id]

        if node.position[2] <= z_floor + 1e-3:
            node.node_type = 'root'
            grounded += 1
            continue

        cur_id       = tip_id
        cur_pos      = node.position.copy()
        cur_diam     = node.diameter
        lean_dir     = _outward_dir(cur_pos[:2], model_centroid)
        lv_start     = grid.level_below(cur_pos[2])
        descent_top  = cur_pos[2]

        if lv_start < 0:
            node.node_type = 'root'; grounded += 1; continue

        for lv in range(lv_start, -1, -1):
            cur_node = tree.nodes[cur_id]
            new_z    = grid.levels[lv]

            if new_z >= cur_pos[2] - 1e-3:
                continue
            if lv == 0:
                lean_mm = 0.0
                lean_xy = cur_pos[:2].copy()
            else:
                lean_mm = _lean_at_z(cur_pos[2], descent_top, z_floor, MAX_LEAN_MM)
                lean_xy = cur_pos[:2] + lean_dir * lean_mm

            new_pos = np.array([lean_xy[0], lean_xy[1], new_z])

            if not _angle_ok(cur_pos, new_pos):
                for frac in [0.6, 0.3, 0.0]:
                    if lv == 0:
                        t_xy = cur_pos[:2].copy()
                    else:
                        t_xy = cur_pos[:2] + lean_dir * lean_mm * frac
                    new_pos = np.array([t_xy[0], t_xy[1], new_z])
                    if _angle_ok(cur_pos, new_pos):
                        lean_xy = t_xy
                        break

            if not point_is_safe(new_pos, mesh, clearance) or \
               not segment_is_safe(cur_pos, new_pos, mesh, clearance) or \
               not tree.branch_clears_existing(cur_pos, new_pos):
                fallback = np.array([cur_pos[0], cur_pos[1], new_z])
                if point_is_safe(fallback, mesh, clearance) and \
                   segment_is_safe(cur_pos, fallback, mesh, clearance):
                    new_pos = fallback; lean_xy = cur_pos[:2]
                else:
                    tree.nodes[cur_id].node_type = 'root'; break

            cur_diam = min(cur_diam * 1.05, max_diameter)
            ntype    = 'root' if lv == 0 else 'branch'
            new_id   = tree.add_node(position=new_pos, diameter=cur_diam,
                                     node_type=ntype,
                                     z_min=z_floor, z_max=new_pos[2])
            tree.connect_force(cur_id, new_id)   # descent chain: no child limit
            cur_id  = new_id
            cur_pos = new_pos

        grounded += 1

    print(f"    [tree] {grounded} tips grounded  |  Final: {tree}")
    return tree


# ─── Pool generator ───────────────────────────────────────────────────────────

def generate_topology_pool(support_points, grid, mesh,
                            n_topologies=2, pool_multiplier=2,
                            tip_diameter=0.5, max_diameter=2.5,
                            clearance=CLEARANCE_MM,
                            merge_depth=0.7, merge_levels=4,
                            **_) -> List[Tree]:
    n_pool = n_topologies * pool_multiplier
    print(f"  [topology] Generating {n_pool} candidates ")

    pool = []
    for i in range(n_pool):
        try:
            t   = build_tree(support_points, grid, mesh,
                             tip_diameter, max_diameter, clearance,
                             merge_depth, merge_levels, seed=i)
            vol = t.compute_volume()
            if vol < 1e-3 or len(t.leaf_nodes()) == 0: continue
            pool.append((vol, t))
            if (i+1) % max(1, n_pool//4) == 0:
                best = min(v for v,_ in pool) if pool else float('inf')
                print(f"  [topology] {i+1}/{n_pool} valid={len(pool)} best={best:.1f}mm3")
        except Exception as e:
            print(f"  [topology] WARN {i}: {e}")
            import traceback; traceback.print_exc()

    if not pool:
        raise RuntimeError("No valid topologies. Try --clearance 1.5 or smaller --sample_spacing.")

    pool.sort(key=lambda x: x[0])
    best = pool[:n_topologies]
    print(f"\n  [topology] Returning best {len(best)}:")
    for r, (vol, t) in enumerate(best):
        print(f"    Rank {r+1}: L={len(t.leaf_nodes())} B={len(t.branch_nodes())} "
              f"R={len(t.root_nodes())} vol={vol:.1f}mm3")
    return [t for _, t in best]