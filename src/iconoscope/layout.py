from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree


def assign_grid(
    coords: np.ndarray,
    grid_cols: int,
    grid_rows: int,
    method: str = "full_grid",
    full_grid_algo: str = "auto",
) -> dict[tuple[int, int], int]:
    N = coords.shape[0]
    n_cells = grid_cols * grid_rows

    if method == "first_come":
        x = coords[:, 0]
        y = coords[:, 1]
        assignments: dict[tuple[int, int], int] = {}
        occupied: set[tuple[int, int]] = set()
        for i in range(min(N, n_cells)):
            a = int(np.clip(np.floor(x[i] * grid_rows), 0, grid_rows - 1))
            b = int(np.clip(np.floor(y[i] * grid_cols), 0, grid_cols - 1))
            if (a, b) not in occupied:
                occupied.add((a, b))
                assignments[(a, b)] = i
        return assignments

    elif method == "full_grid":
        grid_cells = np.array(
            [(r, c) for r in range(grid_rows) for c in range(grid_cols)],
            dtype=np.float32,
        )
        cell_centers = (grid_cells + 0.5) / np.array([[grid_rows, grid_cols]], dtype=np.float32)

        if full_grid_algo == "auto":
            full_grid_algo = "hungarian" if n_cells <= 5000 else "kdtree_greedy"

        if full_grid_algo == "hungarian":
            cost = np.linalg.norm(
                coords[:, np.newaxis, :] - cell_centers[np.newaxis, :, :],
                axis=2,
            )
            row_ind, col_ind = linear_sum_assignment(cost[:n_cells, :n_cells])
            assignments = {}
            for cell_idx, img_idx in zip(col_ind, row_ind):
                r, c = int(grid_cells[cell_idx][0]), int(grid_cells[cell_idx][1])
                assignments[(r, c)] = int(img_idx)
            return assignments

        elif full_grid_algo == "kdtree_greedy":
            kdtree = KDTree(coords)
            assignments = {}
            used = np.zeros(N, dtype=bool)
            for cell_idx in range(n_cells):
                cx, cy = cell_centers[cell_idx]
                # start small, double k only if all candidates are already used
                k = min(N, 20)
                while True:
                    _, idxs = kdtree.query([cx, cy], k=k)
                    idxs = np.atleast_1d(idxs)
                    free = [i for i in idxs if not used[i]]
                    if free:
                        idx = free[0]
                        break
                    k = min(N, k * 4)
                used[idx] = True
                r, c = int(grid_cells[cell_idx][0]), int(grid_cells[cell_idx][1])
                assignments[(r, c)] = int(idx)
            return assignments

        else:
            raise ValueError(f"Unknown full_grid_algo: {full_grid_algo!r}")

    else:
        raise ValueError(f"Unknown layout method: {method!r}")
