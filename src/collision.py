"""
Collision Utilities
-------------------
Helpers for keeping tree branches outside the original model.

Two main jobs:
  1. filter_columns_inside_mesh  — remove grid columns whose XY footprint
     falls inside or too close to the model at overhang height.
     This is what prevents supports from spawning inside the stem.

  2. segment_clears_mesh         — check that the straight line between
     two 3D points does not pass through the model.
     Used when choosing merge-node positions.
"""

import numpy as np


# How far (mm) a node must stay from the model surface.
# Increase if branches still touch the model.
CLEARANCE_MM = 2.5


def _sample_column_positions(col_xy, z_values):
    """Return a (K, 3) array of sample points along a column at given z heights."""
    n = len(z_values)
    pts = np.zeros((n, 3), dtype=float)
    pts[:, 0] = col_xy[0]
    pts[:, 1] = col_xy[1]
    pts[:, 2] = z_values
    return pts


def build_proximity_filter(mesh, grid, clearance=CLEARANCE_MM):
    """
    Return a set of column indices that are SAFE to use (not inside or
    too close to the mesh at any height the support might occupy).

    Uses trimesh.proximity.closest_point for distance queries.
    Falls back gracefully if trimesh is not fully available.

    Parameters
    ----------
    mesh      : trimesh.Trimesh
    grid      : Grid
    clearance : float — minimum allowed distance from model surface (mm)

    Returns
    -------
    safe_cols : set[int]
    """
    try:
        from trimesh import proximity 
    except ImportError:
        print("  [collision] trimesh.proximity not available — skipping column filter")
        return set(range(grid.n_cols))

    safe_cols = set()
    blocked   = 0

    # Sample each column at several z heights to check clearance
    sample_zs = grid.levels[::max(1, grid.n_levels // 6)]   # ~6 samples per column

    print(f"  [collision] Checking {grid.n_cols} columns "
          f"at {len(sample_zs)} heights (clearance={clearance} mm)...")

    for col_idx in range(grid.n_cols):
        xy   = grid.col_positions[col_idx]
        pts  = _sample_column_positions(xy, sample_zs)

        # closest_point returns (closest_pt, distance, triangle_id)
        _, dists, _ = proximity.closest_point(mesh, pts)

        min_dist = float(dists.min())

        # A column is blocked if it is inside OR closer than clearance to surface
        if min_dist < clearance:
            blocked += 1
        else:
            safe_cols.add(col_idx)

    print(f"  [collision] Safe columns: {len(safe_cols)} / {grid.n_cols} "
          f"({blocked} blocked by model)")
    return safe_cols


def segment_clears_mesh(p_start, p_end, mesh, n_samples=8, clearance=CLEARANCE_MM):
    """
    Return True if the straight line from p_start to p_end stays at least
    `clearance` mm from the model surface at all sampled points.

    Parameters
    ----------
    p_start, p_end : np.ndarray (3,)
    mesh           : trimesh.Trimesh
    n_samples      : int — number of intermediate check points
    clearance      : float — required clearance (mm)
    """
    try:
        from trimesh import proximity
    except ImportError:
        return True   # can't check — assume clear

    # Sample along the segment
    ts   = np.linspace(0.0, 1.0, n_samples)
    pts  = np.outer(1.0 - ts, p_start) + np.outer(ts, p_end)

    _, dists, _ = proximity.closest_point(mesh, pts)
    return float(dists.min()) >= clearance


def point_clears_mesh(pt, mesh, clearance=CLEARANCE_MM):
    """Return True if pt is at least clearance mm from the model surface."""
    try:
        from trimesh import proximity
        _, dists, _ = proximity.closest_point(mesh, pt.reshape(1, 3))
        return float(dists[0]) >= clearance
    except Exception:
        return True
