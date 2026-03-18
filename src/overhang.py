"""
Phase 1: Overhang Detection and Support Point Sampling
-------------------------------------------------------
Identifies faces of a mesh that require support from below,
then samples a uniform set of points on those faces.

BUG FIX (v2)
------------
The original code incorrectly flagged the bottom face (sitting on the
build plate at z=0) as an overhang because its normal points straight
down (dot product = 1.0 > cos(45deg) = 0.707).
Fix: exclude any face whose centroid is within 0.5 mm of the model's
lowest z value (the build plate).
"""

import numpy as np

DEFAULT_OVERHANG_ANGLE = 45.0
DEFAULT_SAMPLE_SPACING = 2.5    # mm — tighter default for thin-branch trees
MIN_FACE_AREA          = 1e-6
BUILD_PLATE_TOLERANCE  = 0.5    # mm — faces this close to z_min are excluded


def detect_overhangs(mesh, threshold_deg=DEFAULT_OVERHANG_ANGLE):
    """
    Detect triangular faces that require support from below.

    Parameters
    ----------
    mesh          : trimesh.Trimesh
    threshold_deg : float — overhang threshold in degrees

    Returns
    -------
    overhang_mask : np.ndarray of bool, shape (n_faces,)
    """
    print(f"  [overhang] Computing normals for {len(mesh.faces)} faces "
          f"(threshold={threshold_deg}deg)...")

    normals       = mesh.face_normals          # (N, 3)
    downward      = np.array([0.0, 0.0, -1.0])
    dots          = normals @ downward
    threshold_cos = np.cos(np.radians(threshold_deg))

    overhang_mask = dots > threshold_cos

    # ── FIX: exclude faces sitting on the build plate ─────────────────────────
    z_floor       = float(mesh.bounds[0][2])
    face_verts    = mesh.vertices[mesh.faces]       # (N, 3, 3)
    face_centroid_z = face_verts.mean(axis=1)[:, 2] # (N,)
    on_plate      = face_centroid_z < (z_floor + BUILD_PLATE_TOLERANCE)
    excluded      = overhang_mask & on_plate
    overhang_mask = overhang_mask & ~on_plate

    print(f"  [overhang] dot range: [{dots.min():.3f}, {dots.max():.3f}] "
          f"cos_thresh={threshold_cos:.4f}")
    print(f"  [overhang] Overhang faces detected : {(dots > threshold_cos).sum()}")
    print(f"  [overhang] Excluded (on build plate): {excluded.sum()}  "
          f"(z_floor={z_floor:.2f} mm, tol={BUILD_PLATE_TOLERANCE} mm)")
    print(f"  [overhang] Net overhang faces       : {overhang_mask.sum()}")

    return overhang_mask


def sample_support_points(mesh, overhang_mask, spacing=DEFAULT_SAMPLE_SPACING):
    """
    Uniformly sample attachment points on overhang faces.

    Parameters
    ----------
    mesh         : trimesh.Trimesh
    overhang_mask: bool array, shape (n_faces,)
    spacing      : float — target distance between adjacent points (mm)

    Returns
    -------
    support_points : np.ndarray, shape (N, 3)
    """
    overhang_faces = np.where(overhang_mask)[0]
    print(f"  [sample] Sampling {len(overhang_faces)} overhang faces "
          f"(spacing={spacing} mm)...")

    points      = []
    total_area  = 0.0
    skipped     = 0

    for face_idx in overhang_faces:
        face = mesh.faces[face_idx]
        v0   = mesh.vertices[face[0]]
        v1   = mesh.vertices[face[1]]
        v2   = mesh.vertices[face[2]]

        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        if area < MIN_FACE_AREA:
            skipped += 1
            continue

        total_area += area
        n_samples   = max(1, int(np.round(area / (spacing ** 2))))

        for _ in range(n_samples):
            r1, r2 = np.random.random(), np.random.random()
            if r1 + r2 > 1.0:
                r1, r2 = 1.0 - r1, 1.0 - r2
            r3 = 1.0 - r1 - r2
            points.append(r1 * v0 + r2 * v1 + r3 * v2)

    print(f"  [sample] Total overhang area : {total_area:.2f} mm2")
    print(f"  [sample] Degenerate skipped  : {skipped}")
    print(f"  [sample] Support points      : {len(points)}")

    return np.array(points, dtype=float) if points else np.empty((0, 3))
