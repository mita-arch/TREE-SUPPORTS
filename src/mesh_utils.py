"""
Mesh Utilities — Output Generation
-------------------------------------
Converts the optimized tree structure into a triangulated 3D mesh
and exports STL files.

Each branch in the tree is represented as a frustum (truncated cone):
  - bottom face (child node end): radius = child.diameter / 2
  - top face (parent node end):   radius = parent.diameter / 2
  - When radii are equal, this is a standard cylinder.

The frustum is triangulated using N_SEGMENTS around the circumference.
End caps (top and bottom) are included for watertight geometry.
"""

import numpy as np
import trimesh
from .topology import Tree


N_SEGMENTS = 12    # number of sides around each cylinder/frustum
MIN_LENGTH  = 0.1  # mm — skip branches shorter than this


# ─── Geometry Primitives ──────────────────────────────────────────────────────

def _rotation_matrix_z_to_vec(direction: np.ndarray) -> np.ndarray:
    """
    Compute a rotation matrix R such that R @ [0,0,1] = direction (normalized).
    Uses Rodrigues' formula for numerical stability.
    """
    d = direction / np.linalg.norm(direction)
    z = np.array([0.0, 0.0, 1.0])

    # If already aligned (or anti-aligned) with Z
    dot = np.dot(d, z)
    if dot > 1.0 - 1e-9:
        return np.eye(3)
    if dot < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])   # 180° flip

    v = np.cross(z, d)               # rotation axis
    s = np.linalg.norm(v)            # sin(angle)
    c = dot                          # cos(angle)

    # Skew-symmetric matrix for v
    vx = np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ]
    ])
    R = np.eye(3) + vx + (vx @ vx) * (1.0 - c) / (s * s)
    return R


def make_frustum(p_bottom: np.ndarray, p_top: np.ndarray,
                 r_bottom: float, r_top: float,
                 n_seg: int = N_SEGMENTS):
    """
    Create a triangulated frustum (truncated cone) between two 3D points.

    Parameters
    ----------
    p_bottom : np.ndarray, shape (3,) — start point (child node)
    p_top    : np.ndarray, shape (3,) — end point   (parent node)
    r_bottom : float — radius at p_bottom (mm)
    r_top    : float — radius at p_top    (mm)
    n_seg    : int   — number of circumferential segments

    Returns
    -------
    vertices : np.ndarray, shape (V, 3)
    faces    : np.ndarray, shape (F, 3) — triangle indices
    Returns (None, None) if the branch is degenerate.
    """
    direction = p_top - p_bottom
    length = float(np.linalg.norm(direction))

    if length < MIN_LENGTH:
        return None, None

    r_bottom = max(r_bottom, 0.01)
    r_top    = max(r_top,    0.01)

    R = _rotation_matrix_z_to_vec(direction / length)

    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)

    # Circle rings in local Z-space
    ring_bottom_local = np.column_stack([r_bottom * cos_a, r_bottom * sin_a, np.zeros(n_seg)])
    ring_top_local    = np.column_stack([r_top    * cos_a, r_top    * sin_a, np.full(n_seg, length)])

    # Transform to world space
    ring_bottom = (R @ ring_bottom_local.T).T + p_bottom
    ring_top    = (R @ ring_top_local.T).T    + p_bottom   # p_bottom is the base

    # Cap centers
    center_bottom = p_bottom.copy()
    center_top    = p_top.copy()

    # Vertex layout: [ring_bottom(n_seg) | ring_top(n_seg) | center_bottom | center_top]
    n_base = 2 * n_seg
    vertices = np.vstack([ring_bottom, ring_top, [center_bottom], [center_top]])
    idx_cb = n_base        # center bottom index
    idx_ct = n_base + 1    # center top index

    faces = []

    for i in range(n_seg):
        j = (i + 1) % n_seg

        # Side: two triangles per quad
        #   bottom-left, bottom-right, top-left
        faces.append([i, j, n_seg + i])
        #   bottom-right, top-right, top-left
        faces.append([j, n_seg + j, n_seg + i])

        # Bottom cap (winding: inward normal = downward)
        faces.append([idx_cb, j, i])

        # Top cap (winding: outward normal = upward)
        faces.append([idx_ct, n_seg + i, n_seg + j])

    return vertices, np.array(faces, dtype=np.int32)


# ─── Tree → Mesh ──────────────────────────────────────────────────────────────

def tree_to_mesh(tree: Tree, n_seg: int = N_SEGMENTS) -> trimesh.Trimesh:
    """
    Convert a Tree to a watertight trimesh.Trimesh by generating a
    frustum for every branch (edge) in the tree.

    Parameters
    ----------
    tree  : Tree
    n_seg : int — circumferential segments per frustum

    Returns
    -------
    trimesh.Trimesh (or empty mesh if tree has no edges)
    """
    all_vertices = []
    all_faces    = []
    v_offset     = 0
    n_branches   = 0
    n_skipped    = 0

    for nid, node in tree.nodes.items():
        if node.parent_id < 0:
            continue    # root nodes have no branch below them

        parent = tree.nodes[node.parent_id]

        p_child  = node.position
        p_parent = parent.position
        r_child  = node.diameter   / 2.0
        r_parent = parent.diameter / 2.0

        verts, faces = make_frustum(p_child, p_parent, r_child, r_parent, n_seg)

        if verts is None:
            n_skipped += 1
            continue

        all_vertices.append(verts)
        all_faces.append(faces + v_offset)
        v_offset  += len(verts)
        n_branches += 1

    print(f"  [mesh] Converted {n_branches} branches to frustums ({n_skipped} skipped as degenerate)")

    if not all_vertices:
        print("  [mesh] WARNING: no geometry generated — returning empty mesh")
        return trimesh.Trimesh()

    combined_v = np.vstack(all_vertices)
    combined_f = np.vstack(all_faces)

    print(f"  [mesh] Raw mesh: {len(combined_v)} vertices, {len(combined_f)} triangles")

    mesh = trimesh.Trimesh(vertices=combined_v, faces=combined_f, process=False)
    mesh.merge_vertices()
    mesh.fix_normals()
    mesh.fill_holes()

    print(f"  [mesh] Processed mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  [mesh] Watertight: {mesh.is_watertight}")
    print(f"  [mesh] Volume: {mesh.volume:.2f} mm³")

    return mesh


# ─── STL I/O ──────────────────────────────────────────────────────────────────

def save_stl(mesh: trimesh.Trimesh, path: str, binary: bool = True):
    """
    Export a trimesh.Trimesh to an STL file.

    Parameters
    ----------
    mesh   : trimesh.Trimesh
    path   : str — output file path (e.g. 'output/supports.stl')
    binary : bool — True for binary STL (smaller), False for ASCII
    """
    if mesh is None or len(mesh.faces) == 0:
        print(f"  [stl] WARNING: empty mesh, skipping export to {path}")
        return

    fmt = 'stl' if binary else 'stl_ascii'
    mesh.export(path, file_type=fmt)
    import os
    size_kb = os.path.getsize(path) / 1024
    print(f"  [stl] Saved: {path}  ({len(mesh.faces)} faces, {size_kb:.1f} KB)")


def combine_and_save(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, path: str):
    """
    Combine two meshes and save the result.
    Typically used for (original model) + (support structure).
    """
    combined = trimesh.util.concatenate([mesh_a, mesh_b])
    print(f"  [stl] Combined mesh: {len(combined.faces)} faces")
    save_stl(combined, path)
    return combined
