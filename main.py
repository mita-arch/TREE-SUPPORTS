"""
Tree Support Generator  (v4 — correct, collision-free)
=======================================================

Usage:
  python main.py input.stl [options]

T-shape (40x120x104 mm) recommended:
  python main.py t_shape.stl \\
    --sample_spacing 2 --grid_resolution 6 \\
    --tip_diameter 0.5 --max_diameter 2.5 --clearance 2.0
"""

import argparse, os, sys, time
import numpy as np
import trimesh

from src.overhang   import detect_overhangs, sample_support_points
from src.grid       import Grid
from src.topology   import generate_topology_pool
from src.pso        import optimize_tree
from src.mesh_utils import tree_to_mesh, save_stl, combine_and_save


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate lightweight tree supports (v4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("input_stl")
    p.add_argument("--output_dir",      default="output")

    # Phase 1
    p.add_argument("--overhang_angle",  type=float, default=45.0)
    p.add_argument("--sample_spacing",  type=float, default=2.5,
                   help="mm between attachment tips. 2-3 mm recommended.")

    # Phase 2
    p.add_argument("--grid_resolution", type=float, default=6.0,
                   help="mm between Z-levels for merge quantisation.")

    # Safety
    p.add_argument("--clearance",       type=float, default=2.0,
                   help="Min mm any branch must stay from model surface. "
                        "Increase if branches still clip (try 3.0). "
                        "Decrease only if too many supports are rejected.")

    # Diameter
    p.add_argument("--tip_diameter",    type=float, default=0.5,
                   help="Diameter at the very tip where branch meets overhang (mm).")
    p.add_argument("--max_diameter",    type=float, default=2.5,
                   help="Maximum trunk diameter at build plate (mm).")

    # Pool / quality
    p.add_argument("--n_topologies",    type=int,   default=2,
                   help="How many candidate trees to fully optimise.")
    p.add_argument("--merge_depth",     type=float, default=0.7,
                   help="Fraction of support height in which merges are allowed. "
                        "0.7 = merges in top 70%% of height (higher = more merging). "
                        "Increase toward 1.0 to push merges all the way down. "
                        "Decrease toward 0.3 to only merge near the overhang.")
    p.add_argument("--merge_levels",    type=int,   default=4,
                   help="How many Z levels below the lower tip to try when "
                        "finding a merge point. Higher = more likely to find a "
                        "merge across height gaps. Default 4 = tries up to "
                        "4 x grid_resolution mm below the lower tip.")

    # PSO
    p.add_argument("--n_particles",     type=int,   default=30)
    p.add_argument("--n_iterations",    type=int,   default=300)
    p.add_argument("--verbose_every",   type=int,   default=100)
    p.add_argument("--seed",            type=int,   default=None)
    p.add_argument("--use_boolean",     action="store_true", default=False,
                   help="Boolean union for fully watertight mesh. "
                        "Requires: pip install manifold3d")
    return p


def run_pipeline(args):
    t_total = time.time()

    print("=" * 64)
    print("  TREE SUPPORT GENERATOR  v4  (collision-free, closest-pair merge)")
    print("=" * 64)

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[INFO] Seed: {args.seed}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[LOAD] {args.input_stl}")
    if not os.path.isfile(args.input_stl):
        print(f"[ERROR] Not found: {args.input_stl}"); sys.exit(1)

    raw  = trimesh.load(args.input_stl)
    mesh = (trimesh.util.concatenate(list(raw.dump()))
            if isinstance(raw, trimesh.Scene) else raw)

    print(f"[LOAD] Vertices={len(mesh.vertices):,}  "
          f"Faces={len(mesh.faces):,}  Watertight={mesh.is_watertight}")
    print(f"[LOAD] Bounds: {mesh.bounds[0]}  →  {mesh.bounds[1]}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 1  Overhang detection")
    print(f"{'─'*64}")
    t0 = time.time()

    mask = detect_overhangs(mesh, args.overhang_angle)
    if not mask.any():
        print("[WARN] No overhangs found. Check STL orientation (Z up, base at z=0).")
        sys.exit(0)

    pts = sample_support_points(mesh, mask, args.sample_spacing)
    if len(pts) == 0:
        print("[WARN] No support points sampled."); sys.exit(0)

    print(f"[P1] Done in {time.time()-t0:.2f}s")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 2  Topology generation  (clearance={args.clearance} mm)")
    print(f"{'─'*64}")
    t0 = time.time()

    grid = Grid(mesh.bounds, xy_resolution=args.grid_resolution)
    print(f"[P2] {grid}")

    tree_pool = generate_topology_pool(
        support_points = pts,
        grid           = grid,
        mesh           = mesh,
        n_topologies   = args.n_topologies,
        pool_multiplier= 4,
        tip_diameter   = args.tip_diameter,
        max_diameter   = args.max_diameter,
        clearance      = args.clearance,
        merge_depth    = args.merge_depth,
        merge_levels   = args.merge_levels,
    )
    print(f"[P2] Done in {time.time()-t0:.2f}s")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 3  PSO optimisation")
    print(f"{'─'*64}")

    best_tree, best_vol = None, float('inf')
    t_p3 = time.time()

    for rank, tree in enumerate(tree_pool):
        print(f"\n[P3] Topology {rank+1}/{len(tree_pool)}: {tree}")
        t0 = time.time()
        try:
            opt, vol = optimize_tree(
                tree=tree, mesh=mesh,
                n_particles=args.n_particles,
                n_iterations=args.n_iterations,
                min_diameter=args.tip_diameter,
                max_diameter=args.max_diameter,
                verbose_every=args.verbose_every,
            )
        except Exception as e:
            print(f"  [ERROR] {e}"); import traceback; traceback.print_exc(); continue

        print(f"  vol={vol:.1f} mm3  time={time.time()-t0:.1f}s")
        if vol < best_vol:
            best_vol, best_tree = vol, opt
            print(f"  ✓ New best vol={best_vol:.1f} mm3")

    print(f"\n[P3] Total PSO: {time.time()-t_p3:.1f}s")
    if best_tree is None:
        print("[ERROR] All PSO runs failed."); sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  OUTPUT")
    print(f"{'─'*64}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[OUT] Building support mesh...")
    sup = tree_to_mesh(best_tree)

    sp = os.path.join(args.output_dir, "supports_only.stl")
    cp = os.path.join(args.output_dir, "model_with_supports.stl")
    save_stl(sup, sp)
    combine_and_save(mesh, sup, cp)

    print(f"\n{'='*64}")
    print(f"  DONE  ({time.time()-t_total:.1f}s)")
    print(f"{'='*64}")
    print(f"  Volume  : {best_vol:.1f} mm3")
    print(f"  Tip     : {args.tip_diameter} mm")
    print(f"  Trunk   : {args.max_diameter} mm")
    print(f"  Clearance: {args.clearance} mm")
    print(f"  → {sp}")
    print(f"  → {cp}")
    print("=" * 64)


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
