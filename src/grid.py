"""
Grid G Construction — Phase 2.1
---------------------------------
Discretizes the 3D space below the model's overhang regions
into a regular grid of vertical columns and horizontal levels.

Each grid column is a vertical line segment in XY space.
Each horizontal level is a Z-height slice.
The intersection of a column and a level is a candidate tree node position.

Grid layout:
  - Columns are evenly spaced in the XY plane over the model bounding box
  - Levels are evenly spaced in Z from the build plate (z=0 or z_min) up
    to just below the lowest overhang point
"""

import numpy as np


class Grid:
    """
    Represents the discretization grid G used for tree node placement.

    Attributes
    ----------
    col_positions : np.ndarray, shape (n_cols, 2)
        XY position of each column.
    levels : np.ndarray, shape (n_levels,)
        Z height of each horizontal level.
    """

    def __init__(self, bounds, xy_resolution=10.0, z_resolution=None):
        """
        Parameters
        ----------
        bounds       : array-like, shape (2, 3) — [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        xy_resolution: float — spacing between grid columns in XY (mm)
        z_resolution : float — spacing between levels in Z (mm); defaults to xy_resolution
        """
        bounds = np.array(bounds, dtype=float)
        self.bounds = bounds
        self.xy_resolution = xy_resolution
        self.z_resolution = z_resolution if z_resolution is not None else xy_resolution

        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]

        self.z_floor = zmin          # bottom of grid (build plate level)
        self.z_ceiling = zmax        # top of grid

        # Column XY positions — regular grid covering the bounding box
        # We pad slightly so columns extend beyond the model edges
        pad = xy_resolution
        xs = np.arange(xmin - pad, xmax + pad + xy_resolution, xy_resolution)
        ys = np.arange(ymin - pad, ymax + pad + xy_resolution, xy_resolution)

        self.col_positions = np.array(
            [[x, y] for x in xs for y in ys],
            dtype=float
        )
        self.n_cols = len(self.col_positions)

        # Horizontal levels in Z
        self.levels = np.arange(
            self.z_floor,
            self.z_ceiling + self.z_resolution,
            self.z_resolution
        )
        self.n_levels = len(self.levels)

        print(f"  [grid] XY resolution: {xy_resolution} mm")
        print(f"  [grid] Z resolution:  {self.z_resolution} mm")
        print(f"  [grid] Columns: {self.n_cols} (covering x=[{xmin:.1f},{xmax:.1f}], y=[{ymin:.1f},{ymax:.1f}])")
        print(f"  [grid] Levels:  {self.n_levels} (z=[{self.levels[0]:.1f},{self.levels[-1]:.1f}])")

    # ─── Spatial Queries ───────────────────────────────────────────────────────

    def nearest_column(self, point):
        """
        Return the index of the nearest grid column to a given 2D or 3D point.
        Only the XY coordinates are used.
        """
        xy = np.asarray(point)[:2]
        dists = np.linalg.norm(self.col_positions - xy, axis=1)
        return int(np.argmin(dists))

    def nearest_level(self, z):
        """Return the index of the grid level nearest to z."""
        return int(np.argmin(np.abs(self.levels - z)))

    def level_below(self, z):
        """Return the index of the highest grid level strictly below z, or 0."""
        below = np.where(self.levels < z - 1e-6)[0]
        return int(below[-1]) if len(below) > 0 else 0

    def get_node_position(self, col_idx, level_idx):
        """Return the 3D world position of a grid node."""
        x, y = self.col_positions[col_idx]
        z = self.levels[level_idx]
        return np.array([x, y, z], dtype=float)

    def get_neighbor_columns(self, col_idx, max_neighbors=6):
        """
        Return indices of the nearest neighboring columns (excluding self).
        Used to find candidate merge targets when building tree topologies.

        Parameters
        ----------
        col_idx      : int
        max_neighbors: int — at most 6 per the paper's constraint

        Returns
        -------
        List[int]
        """
        xy = self.col_positions[col_idx]
        dists = np.linalg.norm(self.col_positions - xy, axis=1)
        sorted_idx = np.argsort(dists)
        # skip index 0 (the column itself, dist=0)
        return sorted_idx[1:max_neighbors + 1].tolist()

    def columns_near_points(self, points, radius=None):
        """
        For each XY point, return the nearest column index.
        If radius is given, return only columns within that radius.

        Returns
        -------
        col_indices : List[int]
        """
        if radius is None:
            return [self.nearest_column(p) for p in points]

        result = []
        for p in points:
            xy = np.asarray(p)[:2]
            dists = np.linalg.norm(self.col_positions - xy, axis=1)
            nearby = np.where(dists <= radius)[0]
            if len(nearby) == 0:
                nearby = [np.argmin(dists)]
            result.append(nearby.tolist())
        return result

    def __repr__(self):
        return (
            f"Grid(n_cols={self.n_cols}, n_levels={self.n_levels}, "
            f"xy_res={self.xy_resolution}, z_res={self.z_resolution})"
        )
