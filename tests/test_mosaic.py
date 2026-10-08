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


@pytest.mark.parametrize(
    "image_count, canvas_width, canvas_height, aspect_ratio, expected_cols",
    [
        # square canvas, square thumbnails: 10x10 grid of 100px thumbs
        (100, 1000, 1000, 1.0, 10),
        # wide canvas, square thumbnails: 20x10 grid of 100px thumbs
        (200, 2000, 1000, 1.0, 20),
        # tall canvas, square thumbnails: 10x20 grid of 100px thumbs
        (200, 1000, 2000, 1.0, 10),
        # landscape thumbnails (200x100): 5 cols x 10 rows
        (50, 1000, 1000, 2.0, 5),
        # portrait thumbnails (50x100): 20 cols x 10 rows
        (200, 1000, 1000, 0.5, 20),
        # single image fills the canvas
        (1, 1000, 1000, 1.0, 1),
    ],
)
def test_estimate_columns(
    image_count: int,
    canvas_width: int,
    canvas_height: int,
    aspect_ratio: float,
    expected_cols: int,
) -> None:
    """Total thumbnail area should match canvas area for exact-fit cases."""
    assert (
        mosaic.estimate_columns(image_count, canvas_width, canvas_height, aspect_ratio)
        == expected_cols
    )


def test_estimate_columns_increases_with_image_count() -> None:
    """More images on the same canvas require more (smaller) columns."""
    cols = [mosaic.estimate_columns(n, 1000, 1000, 1.0) for n in (10, 100, 1000)]
    assert cols == sorted(cols)
    assert cols[0] < cols[-1]


@pytest.mark.parametrize(
    "image_count, canvas_width, canvas_height, aspect_ratio, expected",
    [
        (100, 1000, 1000, 1.0, (10, 10)),
        (200, 2000, 1000, 1.0, (10, 20)),
        (50, 1000, 1000, 2.0, (10, 5)),
    ],
)
def test_best_grid(
    image_count: int,
    canvas_width: int,
    canvas_height: int,
    aspect_ratio: float,
    expected: tuple[int, int],
) -> None:
    layout = mosaic.best_grid(image_count, canvas_width, canvas_height, aspect_ratio)
    assert (layout.rows, layout.cols) == expected
    assert layout.rows * layout.cols >= image_count
    # ideal thumbnail size preserves aspect ratio and fits within the canvas
    assert layout.thumb_width / layout.thumb_height == pytest.approx(aspect_ratio)
    assert layout.thumb_width * layout.cols <= canvas_width
    assert layout.thumb_height * layout.rows <= canvas_height
