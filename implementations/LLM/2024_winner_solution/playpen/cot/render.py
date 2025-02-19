import base64
from io import BytesIO

import attrs
import numpy as np
from devtools import debug
from PIL import Image

from src.prompts.colors import color_map

# Define the exact color scheme (0-9) as RGB tuples


edge_color = (85, 85, 85)  # Grey edge color
white = (255, 255, 255)  # White

highlight_color = (255, 0, 0)  # Red


@attrs.frozen
class RenderArgs:
    cell_size: int = 40
    use_border: bool = False
    use_larger_edges: bool = True
    use_alt_color_scheme: bool = False
    force_high_res: bool = False
    force_edge_size: int | None = None
    lower_cell_size_on_bigger_to: int | None = None
    # avoid_edge_around_border: bool = False


def create_rgb_grid(
    grid: np.ndarray,
    render_args: RenderArgs = RenderArgs(),
    should_highlight: np.ndarray | None = None,
    lower_right_triangle: np.ndarray | None = None,
):
    height, width = grid.shape

    cell_size = render_args.cell_size
    use_border = render_args.use_border
    use_larger_edges = render_args.use_larger_edges
    force_edge_size = render_args.force_edge_size
    # avoid_edge_around_border = render_args.avoid_edge_around_border

    if render_args.lower_cell_size_on_bigger_to is not None and (
        height > 10 or width > 10
    ):
        cell_size = render_args.lower_cell_size_on_bigger_to

    if force_edge_size is not None:
        edge_size = force_edge_size
    else:
        edge_size = max(cell_size // 8, 1) if use_larger_edges else 1

    # Calculate the size of the new grid with edges
    new_height = height * (cell_size + edge_size) + edge_size
    new_width = width * (cell_size + edge_size) + edge_size

    # Create a new grid filled with the edge color
    rgb_grid = np.full((new_height, new_width, 3), edge_color, dtype=np.uint8)

    # Fill in the cells with the appropriate colors
    for i in range(height):
        for j in range(width):
            color = color_map[grid[i, j]].hex
            start_row = i * (cell_size + edge_size) + edge_size
            start_col = j * (cell_size + edge_size) + edge_size

            if should_highlight is not None and should_highlight[i, j]:
                rgb_grid[
                    start_row : start_row + cell_size, start_col : start_col + cell_size
                ] = highlight_color
                highlight_width = cell_size // 8
                rgb_grid[
                    start_row + highlight_width : start_row
                    + cell_size
                    - highlight_width,
                    start_col + highlight_width : start_col
                    + cell_size
                    - highlight_width,
                ] = color

                assert (
                    lower_right_triangle is None
                ), "Can't highlight and lower right triangle at the same time (yet)"

            else:
                rgb_grid[
                    start_row : start_row + cell_size, start_col : start_col + cell_size
                ] = color

                if lower_right_triangle is not None:
                    lower_right_triangle_color = color_map[
                        lower_right_triangle[i, j]
                    ].hex
                    for r in range(cell_size):
                        for c in range(cell_size):
                            if r > c:
                                rgb_grid[
                                    start_row + r, start_col + cell_size - 1 - c
                                ] = lower_right_triangle_color

    # if avoid_edge_around_border:
    #     return rgb_grid[
    #         edge_size : new_height - edge_size, edge_size : new_width - edge_size
    #     ]

    if not use_border:
        return rgb_grid

    rgb_grid_border = np.full(
        (new_height + cell_size, new_width + cell_size, 3), white, dtype=np.uint8
    )
    assert cell_size % 2 == 0
    rgb_grid_border[
        cell_size // 2 : new_height + cell_size // 2,
        cell_size // 2 : new_width + cell_size // 2,
    ] = rgb_grid

    return rgb_grid_border


def grid_to_pil(
    grid: np.ndarray,
    render_args: RenderArgs = RenderArgs(),
    should_highlight: np.ndarray | None = None,
    lower_right_triangle: np.ndarray | None = None,
):
    rgb_grid = create_rgb_grid(
        grid,
        render_args=render_args,
        should_highlight=should_highlight,
        lower_right_triangle=lower_right_triangle,
    )
    return Image.fromarray(rgb_grid, "RGB")


def grid_to_base64_png(
    grid: np.ndarray,
    render_args: RenderArgs = RenderArgs(),
    should_highlight: np.ndarray | None = None,
    lower_right_triangle: np.ndarray | None = None,
) -> str:
    image = grid_to_pil(
        grid,
        render_args=render_args,
        should_highlight=should_highlight,
        lower_right_triangle=lower_right_triangle,
    )

    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("utf-8")


def grid_to_base64_png_oai_content(
    grid: np.ndarray,
    render_args: RenderArgs = RenderArgs(),
    should_highlight: np.ndarray | None = None,
    lower_right_triangle: np.ndarray | None = None,
):
    base64_png = grid_to_base64_png(
        grid,
        render_args=render_args,
        should_highlight=should_highlight,
        lower_right_triangle=lower_right_triangle,
    )

    # rgb_grid_for_shape = create_rgb_grid(
    #     grid,
    #     render_args=render_args,
    #     should_highlight=should_highlight,
    #     lower_right_triangle=lower_right_triangle,
    # )

    extra = {"detail": "high"} if render_args.force_high_res else {}

    # print(f"{rgb_grid_for_shape.shape=}")

    # NOTE: we currently use "auto". Seems fine for now I think...
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_png}",
            **extra,
        },
    }


def show_grid(
    grid: np.ndarray,
    render_args: RenderArgs = RenderArgs(),
    should_highlight: np.ndarray | None = None,
    lower_right_triangle: np.ndarray | None = None,
):
    grid_to_pil(
        grid,
        render_args=render_args,
        should_highlight=should_highlight,
        lower_right_triangle=lower_right_triangle,
    ).show()


# # Example usage
# initial_values = np.array([[1, 1, 2], [2, 3, 5], [0, 2, 1]])

# rgb_grid = create_rgb_grid(initial_values, cell_size=10)

# image = Image.fromarray(rgb_grid, "RGB")
# image.show()
