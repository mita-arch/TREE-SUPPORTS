"""
Tree Support Generator  (v3 — collision-aware, outward lean)
=============================================================

Usage:
  python main.py input.stl [options]

T-shape recommended command:
  python main.py t_shape.stl \\
      --sample_spacing 2 --grid_resolution 6 \\
      --tip_diameter 0.5 --max_diameter 2.5 \\
      --n_merge_passes 5 --clearance 2.5
"""

import argparse, os, sys, time
import numpy as np
import trimesh

from src.overhang   import detect_overhangs, sample_support_points
from src.grid       import Grid
from src.collision  import build_proximity_filter
from src.topology   import generate_topology_pool
from src.pso        import optimize_tree
from src.mesh_utils import tree_to_mesh, save_stl, combine_and_save


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate lightweight tree supports for 3D printing (v3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("input_stl")
    p.add_argument("--output_dir",      default="output")

    # Phase 1
    p.add_argument("--overhang_angle",  type=float, default=45.0)
    p.add_argument("--sample_spacing",  type=float, default=2.5,
                   help="mm between attachment points. Smaller = more thin tips.")

    # Phase 2 — geometry
    p.add_argument("--grid_resolution", type=float, default=7.0,
                   help="mm between grid columns. Smaller = finer branching.")
    p.add_argument("--n_topologies",    type=int,   default=2)
    p.add_argument("--n_merge_passes",  type=int,   default=5,
                   help="Descend+merge cycles. More = fewer trunks, more merging.")

    # Collision / clearance
    p.add_argument("--clearance",       type=float, default=2.5,
                   help="Min mm between any support branch and the model surface.")

    # Diameter
    p.add_argument("--tip_diameter",    type=float, default=0.5,
                   help="Diameter at tip where branch meets model (mm).")
    p.add_argument("--max_diameter",    type=float, default=2.5,
                   help="Max trunk diameter at build plate (mm).")

    # PSO
    p.add_argument("--n_particles",     type=int,   default=30)
    p.add_argument("--n_iterations",    type=int,   default=300)
    p.add_argument("--verbose_every",   type=int,   default=100)
    p.add_argument("--seed",            type=int,   default=None)
    return p


def run_pipeline(args):
    t_total = time.time()

    print("=" * 64)
    print("  TREE SUPPORT GENERATOR  v3  (collision-aware)")
    print("=" * 64)

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[INFO] Seed: {args.seed}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[LOAD] {args.input_stl}")
    if not os.path.isfile(args.input_stl):
        print(f"[ERROR] Not found: {args.input_stl}")
        sys.exit(1)

    raw = trimesh.load(args.input_stl)
    mesh = (trimesh.util.concatenate(list(raw.dump()))
            if isinstance(raw, trimesh.Scene) else raw)

    print(f"[LOAD] Vertices={len(mesh.vertices):,}  "
          f"Faces={len(mesh.faces):,}  "
          f"Watertight={mesh.is_watertight}")
    print(f"[LOAD] Bounds: {mesh.bounds[0]} → {mesh.bounds[1]}")

    # Model XY centroid used for outward lean direction
    model_center_xy = np.array([
        (mesh.bounds[0][0] + mesh.bounds[1][0]) / 2.0,
        (mesh.bounds[0][1] + mesh.bounds[1][1]) / 2.0,
    ])
    print(f"[LOAD] Model XY centroid (lean origin): {model_center_xy}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 1  Overhang detection & sampling")
    print(f"{'─'*64}")
    t0 = time.time()

    overhang_mask = detect_overhangs(mesh, args.overhang_angle)
    if not overhang_mask.any():
        print("[WARN] No overhangs found. Check STL orientation (Z up, base at z=0).")
        sys.exit(0)

    support_points = sample_support_points(mesh, overhang_mask, args.sample_spacing)
    if len(support_points) == 0:
        print("[WARN] No support points generated.")
        sys.exit(0)

    print(f"[P1] Done in {time.time()-t0:.2f}s")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 2  Grid + collision filter + topology pool")
    print(f"{'─'*64}")
    t0 = time.time()

    grid = Grid(mesh.bounds, xy_resolution=args.grid_resolution)
    print(f"[P2] Grid: {grid}")

    print(f"\n[P2] Filtering grid columns (clearance={args.clearance} mm)...")
    from src.collision import CLEARANCE_MM
    # Temporarily override module-level clearance with CLI value
    import src.collision as _coll
    _coll.CLEARANCE_MM = args.clearance

    safe_cols = build_proximity_filter(mesh, grid, clearance=args.clearance)

    if len(safe_cols) == 0:
        print("[ERROR] All grid columns blocked by model. "
              "Try --clearance smaller (e.g. 1.0) or --grid_resolution smaller.")
        sys.exit(1)

    print(f"\n[P2] Building topology pool...")
    tree_pool = generate_topology_pool(
        support_points  = support_points,
        grid            = grid,
        mesh            = mesh,
        safe_cols       = safe_cols,
        model_center_xy = model_center_xy,
        n_topologies    = args.n_topologies,
        pool_multiplier = 5,
        tip_diameter    = args.tip_diameter,
        max_diameter    = args.max_diameter,
        n_merge_passes  = args.n_merge_passes,
    )
    print(f"[P2] Done in {time.time()-t0:.2f}s")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 3  PSO optimisation")
    print(f"{'─'*64}")

    best_tree = None
    best_vol  = float('inf')
    t_p3      = time.time()

    for rank, tree in enumerate(tree_pool):
        print(f"\n[P3] Topology {rank+1}/{len(tree_pool)}: {tree}")
        t0 = time.time()
        try:
            opt_tree, opt_vol = optimize_tree(
                tree=tree, mesh=mesh,
                n_particles=args.n_particles,
                n_iterations=args.n_iterations,
                min_diameter=args.tip_diameter,
                max_diameter=args.max_diameter,
                verbose_every=args.verbose_every,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            continue

        print(f"  Result: vol={opt_vol:.1f} mm3  time={time.time()-t0:.1f}s")
        if opt_vol < best_vol:
            best_vol  = opt_vol
            best_tree = opt_tree
            print(f"  ✓ New best vol={best_vol:.1f} mm3")

    print(f"\n[P3] PSO time: {time.time()-t_p3:.1f}s")

    if best_tree is None:
        print("[ERROR] All PSO runs failed.")
        sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  OUTPUT")
    print(f"{'─'*64}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[OUT] Building support mesh...")
    support_mesh = tree_to_mesh(best_tree)

    sp = os.path.join(args.output_dir, "supports_only.stl")
    cp = os.path.join(args.output_dir, "model_with_supports.stl")
    save_stl(support_mesh, sp)
    combine_and_save(mesh, support_mesh, cp)

    print(f"\n{'='*64}")
    print(f"  DONE  ({time.time()-t_total:.1f}s total)")
    print(f"{'='*64}")
    print(f"  Support volume : {best_vol:.1f} mm3")
    print(f"  Tip diameter   : {args.tip_diameter} mm")
    print(f"  Max diameter   : {args.max_diameter} mm")
    print(f"  Clearance      : {args.clearance} mm")
    print(f"  supports_only.stl        → {sp}")
    print(f"  model_with_supports.stl  → {cp}")
    print("=" * 64)


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
