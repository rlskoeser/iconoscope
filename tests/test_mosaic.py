import numpy as np
import pytest

from iconoscope import mosaic


@pytest.mark.parametrize("grid_cols, grid_rows", [(2, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("use_greedy", [False, True])
def test_assign_to_grid_preserves_xy_orientation(
    monkeypatch: pytest.MonkeyPatch,
    grid_cols: int,
    grid_rows: int,
    use_greedy: bool,
) -> None:
    """Coordinates are x/y, while assignments are returned as row/column."""
    if use_greedy:
        monkeypatch.setattr(mosaic, "LAPJV_CELL_LIMIT", 0)

    coords = np.array(
        [
            (col + 0.5, row + 0.5)
            for row in range(grid_rows)
            for col in range(grid_cols)
        ],
        dtype=np.float32,
    )
    coords /= (grid_cols, grid_rows)

    assignments = mosaic.assign_to_grid(coords, grid_cols, grid_rows)

    assert assignments == {
        (row, col): row * grid_cols + col
        for row in range(grid_rows)
        for col in range(grid_cols)
    }
