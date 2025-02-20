from collections import defaultdict

import numpy as np

from src.models import GRID
from src.prompts.colors import color_map

spreadsheet_col_labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "AA",
    "AB",
    "AC",
    "AD",
]


def spreadsheet_ascii_grid(grid: np.ndarray, separator: str = "|") -> str:
    rows, cols = grid.shape
    assert cols <= 30
    assert rows <= 30

    cols_header_line = separator.join([" "] + spreadsheet_col_labels[:cols])
    rest = "\n".join(
        separator.join([str(i + 1)] + [str(x) for x in row])
        for i, row in enumerate(grid)
    )

    return f"{cols_header_line}\n{rest}"


def grid_to_ascii(
    grid: np.ndarray, separator: str = "|", spreadsheet_ascii: bool = False
) -> str:
    if spreadsheet_ascii:
        return spreadsheet_ascii_grid(grid, separator=separator)

    return "\n".join(separator.join(str(x) for x in row) for row in grid)


def array_to_str(grid: GRID) -> str:
    final_output = "["
    for i, row in enumerate(grid):
        final_output += f"{str(row)}"
        if i != len(grid) - 1:
            final_output += ","
    final_output += "]"
    return final_output


def grid_diffs_to_ascii(
    grid_input: np.ndarray, grid_output: np.ndarray, separator: str = "|"
) -> str:
    assert grid_input.shape == grid_output.shape
    row_nums = grid_input.shape[0]
    col_nums = grid_input.shape[1]
    diff_arr: list[list[str]] = [
        ["  --  " for col in range(col_nums)] for row in range(row_nums)
    ]
    # iterate through the rows and columns using the shape
    for row_ind in range(row_nums):
        for col_ind in range(col_nums):
            if grid_input[row_ind][col_ind] != grid_output[row_ind][col_ind]:
                diff_arr[row_ind][col_ind] = (
                    f"{grid_input[row_ind][col_ind]} -> {grid_output[row_ind][col_ind]}"
                )

    return "\n".join(separator.join(str(x) for x in row) for row in diff_arr)


def get_spreadsheet_notation_str(i, j, quote: bool = True):
    out = f"{spreadsheet_col_labels[j]}{i+1}"
    if quote:
        out = f'"{out}"'
    return out


def get_spreadsheet_notation_support_runs(rows_cols: list[tuple[int, int]]):
    row_cols_v = np.array(sorted(rows_cols, key=lambda x: (x[0], x[1])))

    running_str = ""

    idx = 0
    while idx < len(row_cols_v):
        r, c = row_cols_v[idx]

        count_in_a_row = 0
        for checking_idx, (n_r, n_c) in enumerate(row_cols_v[idx:]):
            if n_r == r and n_c == c + checking_idx:
                count_in_a_row += 1
            else:
                break

        if count_in_a_row > 4:
            start = get_spreadsheet_notation_str(r, c, quote=False)
            c_end = c + count_in_a_row - 1

            assert np.array_equal(row_cols_v[idx + count_in_a_row - 1], (r, c_end)), (
                row_cols_v[idx + count_in_a_row - 1],
                (r, c_end),
            )

            end = get_spreadsheet_notation_str(r, c_end, quote=False)

            running_str += f" {start} ... {end}"
            idx += count_in_a_row
        else:
            running_str += " " + get_spreadsheet_notation_str(r, c, quote=False)
            idx += 1

    return running_str


def spreadsheet_ascii_grid_by_color_diffs(
    grid_input: np.ndarray,
    grid_output: np.ndarray,
    use_alt_color_scheme: bool = True,
    use_expected_vs_got: bool = False,
) -> str:
    assert grid_input.shape == grid_output.shape
    grid_differs_x, grid_differs_y = (grid_input != grid_output).nonzero()
    differences_by_color_pairs: dict[tuple[int, int], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    for x, y in zip(grid_differs_x.tolist(), grid_differs_y.tolist(), strict=False):
        differences_by_color_pairs[(grid_input[x, y], grid_output[x, y])].append(
            (int(x), int(y))
        )
    out = ""
    for (color_input, color_output), differing_locs in sorted(
        differences_by_color_pairs.items(), key=lambda x: x[0]
    ):
        color_str = get_spreadsheet_notation_support_runs(differing_locs)

        if use_expected_vs_got:
            out += (
                f"Expected {color_map[color_input].name} ({color_input}) but got {color_map[color_output].name} ({color_output}):{color_str}"
            ) + "\n"

        else:
            out += (
                f"{color_map[color_input].name} ({color_input}) to {color_map[color_output].name} ({color_output}):{color_str}"
            ) + "\n"
    return out


# import attrs


# @attrs.frozen
class StdoutStderr:
    stdout: str
    stderr: str


# @attrs.frozen
class RenderArgs:
    cell_size: int = 40
    use_border: bool = False
    use_larger_edges: bool = True
    use_alt_color_scheme: bool = False
    force_high_res: bool = False
    force_edge_size: int | None = None
    lower_cell_size_on_bigger_to: int | None = None
    # avoid_edge_around_border: bool = False


# @attrs.frozen
class DisplayArgs:
    render_args: RenderArgs = RenderArgs()
    use_diff_highlight: bool = False
    use_diff_triangles: bool = False
    ascii_separator: str = "|"
    spreadsheet_ascii: bool = False
    spreadsheet_ascii_full: bool = False
    spreadsheet_ascii_show_diff_if_concise: bool = False
    hacky_allow_size_mismatch_input_output: bool = False
    disable_absolute_in_normalized_ascii: bool = False
    max_allowed_tokens_per_color: int | None = 200
    max_allowed_tokens_full_ascii_grid: int | None = None
    connected_include_diagonals: bool = False


def display_wrong_output_alt(
    item_idx: int,
    item: list[list[int]] | None,
    expected_output: list[list[int]],
    stdout_stderr: StdoutStderr,
    display_args: DisplayArgs = DisplayArgs(),
    use_output_diff: bool = True,
):
    expected_shape = np.array(expected_output).shape
    x_expected, y_expected = expected_shape

    fmt_num = item_idx + 1

    basic_title = f"# Example {fmt_num}\n\n## Output for Example {fmt_num} from the incorrect `transform` function (aka actual output)\n\n"

    if stdout_stderr.stdout == "" and stdout_stderr.stderr == "":
        stdout_stderr_text = " stdout and stderr were empty."
    else:
        stdout_stderr_text = f"\n\nHere are the stdout and stderr of the function for this example:\n\n<stdout>\n{stdout_stderr.stdout}\n</stdout>\n\n<stderr>{stdout_stderr.stderr}</stderr>"

    if item == expected_output:
        return [
            {
                "type": "text",
                "text": basic_title
                + f"The output matches the expected output. (It is correct.){stdout_stderr_text}\n\n",
            }
        ]

    if item is None:
        return [
            {
                "type": "text",
                "text": basic_title
                + f"There was an error when running the function on this input.{stdout_stderr_text}\n\n",
            }
        ]

    has_some_invalid = any(not (0 <= x <= 9) for row in item for x in row)

    invalid_text = "Note that the output contains some invalid values (values that are not between 0 and 9 inclusive). These invalid values are incorrect and will need to be fixed. Invalid values are displayed in white in the image representation and the actual (invalid) value is displayed in the ASCII representation.\n\n"
    if not has_some_invalid:
        invalid_text = ""

    actual_shape = np.array(item).shape

    # TODO: diff text!!!

    grid_expected = np.array(expected_output)
    grid_actual = np.array(item)

    out = [
        {
            "type": "text",
            "text": basic_title + invalid_text + stdout_stderr_text.strip() + "\n\n",
        },
        *display_single_grid_alt(
            item,
            display_args=display_args,
            extra_shape_text=(
                f" (Shape differs from expected shape. The expected shape is: {x_expected} by {y_expected}.)"
                if actual_shape != expected_shape
                else ""
            ),
        ),
        {
            "type": "text",
            "text": "## Expected Output\n\n",
        },
        *display_single_grid_alt(
            expected_output,
            display_args=display_args,
        ),
    ]

    if use_output_diff:
        if grid_expected.shape != grid_actual.shape:
            color_changes = " [OMITTED DUE TO SHAPE MISMATCH]"
        elif not diff_is_concise(grid_input=grid_expected, grid_output=grid_actual):
            color_changes = " [OMITTED DUE TO EXCESSIVE LENGTH]"
        else:
            color_changes = spreadsheet_ascii_grid_by_color_diffs(
                grid_input=grid_expected,
                grid_output=grid_actual,
                use_alt_color_scheme=display_args.render_args.use_alt_color_scheme,
                use_expected_vs_got=True,
            )

        diff_rep_for_actual_expected = f"""## Color differences between the expected output and the actual output\n\n{color_changes}\n\n"""
        out.append(
            {
                "type": "text",
                "text": diff_rep_for_actual_expected,
            },
        )

    return out


if __name__ == "__main__":
    g = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    out = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    print(grid_to_ascii(g, spreadsheet_ascii=False))
    # print(spreadsheet_ascii_grid_by_color_diffs(grid_input=g, grid_output=out))
    print(grid_diffs_to_ascii(grid_input=g, grid_output=out))
