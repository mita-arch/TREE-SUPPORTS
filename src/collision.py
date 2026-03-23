"""
Collision Utilities  (v4 — raycasting inside/outside + segment sampling)
------------------------------------------------------------------------

Two distinct tests are needed:

  1. inside_mesh(pts, mesh)
     Uses ray casting (trimesh.ray) to determine whether points are
     geometrically INSIDE the solid model.  This is the correct test
     for "is a node placed inside geometry" — proximity to surface is
     NOT sufficient because a point can be far from any surface yet
     still be inside a convex region.

  2. segment_clears_mesh(a, b, mesh, clearance)
     Samples N points along the line a→b and checks each one is:
       (a) not inside the mesh, AND
       (b) at least `clearance` mm from any surface.
     Used for every branch before it is placed.
"""

import numpy as np

CLEARANCE_MM  = 2.0   # min distance from model surface
N_SEG_SAMPLES = 10    # samples per branch segment for clearance check


# ─── Inside/outside test ──────────────────────────────────────────────────────

def points_inside_mesh(pts: np.ndarray, mesh) -> np.ndarray:
    """
    Return a bool array: True where pts[i] is INSIDE the mesh volume.

    Uses winding-number / ray-intersection via trimesh.  Falls back to
    all-False (assume outside) if trimesh ray is unavailable.

    Parameters
    ----------
    pts  : (N, 3) array of query points
    mesh : trimesh.Trimesh

    Returns
    -------
    inside : (N,) bool array
    """
    pts = np.atleast_2d(pts).astype(float)
    try:
        return mesh.contains(pts)
    except Exception:
        return np.zeros(len(pts), dtype=bool)


def point_is_inside(pt: np.ndarray, mesh) -> bool:
    """Scalar version of points_inside_mesh."""
    return bool(points_inside_mesh(pt.reshape(1, 3), mesh)[0])


# ─── Surface proximity test ───────────────────────────────────────────────────

def points_near_surface(pts: np.ndarray, mesh, clearance: float = CLEARANCE_MM
                        ) -> np.ndarray:
    """
    Return bool array: True where pts[i] is closer than `clearance` to
    any surface triangle (regardless of inside/outside).

    Parameters
    ----------
    pts      : (N, 3)
    mesh     : trimesh.Trimesh
    clearance: float — mm

    Returns
    -------
    too_close : (N,) bool array
    """
    pts = np.atleast_2d(pts).astype(float)
    try:
        from trimesh import proximity
        _, dists, _ = proximity.closest_point(mesh, pts)
        return dists < clearance
    except Exception:
        return np.zeros(len(pts), dtype=bool)


# ─── Point safety (combined) ──────────────────────────────────────────────────

def point_is_safe(pt: np.ndarray, mesh, clearance: float = CLEARANCE_MM) -> bool:
    """
    Return True if pt is:
      (a) NOT inside the mesh, AND
      (b) at least `clearance` mm from any surface.

    This is the single gate that every candidate node must pass.
    """
    p = pt.reshape(1, 3)
    if point_is_inside(pt, mesh):
        return False
    near = points_near_surface(p, mesh, clearance)
    return not bool(near[0])


# ─── Segment safety ───────────────────────────────────────────────────────────

def segment_is_safe(p_start: np.ndarray, p_end: np.ndarray,
                    mesh, clearance: float = CLEARANCE_MM,
                    n_samples: int = N_SEG_SAMPLES) -> bool:
    """
    Return True if the straight line p_start → p_end is entirely safe
    (no sampled point is inside the mesh or closer than clearance to surface).

    Parameters
    ----------
    p_start, p_end : (3,) arrays
    mesh           : trimesh.Trimesh
    clearance      : float — mm
    n_samples      : int — number of intermediate check points
    """
    ts   = np.linspace(0.05, 0.95, n_samples)
    pts  = np.outer(1.0 - ts, p_start) + np.outer(ts, p_end)  # (N, 3)

    # Inside check
    inside = points_inside_mesh(pts, mesh)
    if inside.any():
        return False

    # Proximity check
    near = points_near_surface(pts, mesh, clearance)
    if near.any():
        return False

    return True


# ─── Segment-segment minimum distance ────────────────────────────────────────

def seg_seg_min_dist(p1: np.ndarray, p2: np.ndarray,
                     p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Minimum distance between two finite line segments p1-p2 and p3-p4.
    Used to detect branch-branch collisions.
    """
    d1  = p2 - p1
    d2  = p4 - p3
    r   = p1 - p3
    a   = np.dot(d1, d1)
    e   = np.dot(d2, d2)
    f   = np.dot(d2, r)

    if a < 1e-10 and e < 1e-10:
        return float(np.linalg.norm(r))

    if a < 1e-10:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e < 1e-10:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b    = np.dot(d1, d2)
            denom = a * e - b * b
            s    = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-10 else 0.0
            t    = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    return float(np.linalg.norm(closest1 - closest2))
