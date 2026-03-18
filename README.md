<<<<<<< HEAD
# Tree Support Generator

Generates lightweight tree-shaped 3D printing supports for overhanging regions of an STL model.

Based on the algorithm from:
> *"Lightweight Tree Supports for 3D Printing"*
> Computer-Aided Design & Applications, 17(4), 2020, 716-726

---

## Algorithm Overview

| Phase | What happens |
|-------|-------------|
| **Phase 1** | Detect overhang faces (tilt > threshold) → sample uniform support points |
| **Phase 2** | Build a 3D grid G → generate `10×I` random tree topologies → keep best `I` |
| **Phase 3** | PSO + greedy optimization on each topology → pick globally best tree |
| **Output**  | `supports_only.stl` and `model_with_supports.stl` |

---

## Installation

```bash
# Python 3.8+ required

# 1. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate.bat       # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py <input.stl> [options]
```

### Minimal example
```bash
python main.py my_model.stl
```

Outputs are written to `./output/` by default:
- `output/supports_only.stl`
- `output/model_with_supports.stl`

### Full options
```bash
python main.py my_model.stl \
    --output_dir      ./results \
    --overhang_angle  45         \   # degrees — faces tilted more than this need support
    --sample_spacing  5          \   # mm between support sample points
    --grid_resolution 10         \   # mm between grid columns in XY
    --n_topologies    3          \   # I — number of topologies to PSO-optimize
    --n_particles     30         \   # PSO swarm size (paper: 100)
    --n_iterations    300        \   # PSO iterations  (paper: 2000)
    --min_diameter    1.0        \   # minimum branch diameter (mm)
    --max_diameter    4.0        \   # maximum branch diameter (mm)
    --seed            42         \   # for reproducibility
    --verbose_every   50             # print PSO progress every N iterations
```

### Higher-quality output (slower)
```bash
python main.py my_model.stl \
    --n_topologies 5 --n_particles 100 --n_iterations 2000
```

---

## Output Files

| File | Contents |
|------|----------|
| `supports_only.stl` | Only the tree support structure |
| `model_with_supports.stl` | Original model + supports combined |

Both are binary STL files ready for slicing.

---

## Parameters Guide

### `--overhang_angle` (default: 45°)
- The maximum self-supporting angle for your printer (typically 45°).
- Increase to generate supports only for very steep overhangs.
- Decrease to generate more conservative/cautious support coverage.

### `--sample_spacing` (default: 5 mm)
- Distance between adjacent support attachment points.
- Smaller → more support points → denser, stronger supports (more material).
- Larger → fewer points → lighter supports (may leave gaps).

### `--grid_resolution` (default: 10 mm)
- Spacing between grid columns in XY.
- Smaller → finer branching structure → more realistic tree shape, slower.
- Larger → coarser branching → faster but less optimal.

### `--n_topologies` (default: 3)
- How many candidate tree topologies to fully optimize with PSO.
- Increasing this improves solution quality at a linear cost in time.

### `--n_particles` / `--n_iterations`
- Controls PSO quality. More = better solution, slower runtime.
- Paper values: 100 particles × 2000 iterations.
- For a quick test: 20 particles × 100 iterations.

---

## Project Structure

```
tree_support/
├── main.py                  # CLI entry point & pipeline orchestration
├── requirements.txt         # Python dependencies
├── README.md
└── src/
    ├── __init__.py
    ├── overhang.py          # Phase 1: overhang detection + point sampling
    ├── grid.py              # Phase 2.1: Grid G construction
    ├── topology.py          # Phase 2.2: tree data structure + topology generation
    ├── pso.py               # Phase 3: PSO optimizer + greedy surface attachment
    └── mesh_utils.py        # Frustum geometry generation + STL I/O
```

---

## Troubleshooting

**"No overhangs detected"**
- Check that your STL is oriented with the build plate at Z=0 (Z-up).
- Try increasing `--overhang_angle` (e.g., `--overhang_angle 60`).

**"No topology pool generated"**
- Try reducing `--grid_resolution` (e.g., `5`) or `--sample_spacing` (e.g., `3`).

**Supports look too sparse**
- Reduce `--sample_spacing`.

**Supports look too dense / too much material**
- Increase `--sample_spacing` or `--grid_resolution`.

**Very slow runtime**
- Reduce `--n_particles` and `--n_iterations` for a quick preview.
- Reduce `--n_topologies` to 1.
=======
# TREE-SUPPORTS
>>>>>>> c5572b4e0c84414d3c1c54b52ec260485b1000d9
