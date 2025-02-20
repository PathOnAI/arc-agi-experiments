from render import alt_color_scheme_consts_name, color_scheme_consts_name


def get_system_prompt(
    use_diff_highlight: bool = False,
    use_diff_triangles: bool = False,
    additional_info: bool = True,
    just_reasoning_additional_info: bool = True,
    just_attributes_additional_info: bool = False,
    use_many_ascii_representations: bool = False,
    use_alt_color_scheme: bool = False,
    disable_absolute_in_normalized_ascii: bool = False,
    long_as_you_want: bool = False,
    use_diff_rep: bool = False,
    use_resolve_ambiguity: bool = True,
    use_multi_part_transformation_rule_hint: bool = False,
    use_explain_connected: bool = False,
    connected_include_diagonals: bool = False,
) -> str:
    scheme = (
        alt_color_scheme_consts_name
        if use_alt_color_scheme
        else color_scheme_consts_name
    )

    color_to_index = ", ".join(
        f"{name}: {color_val}" for color_val, name in enumerate(scheme.values())
    )

    many_ascii_rep_and_image_version_of_input_line = f"""The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as images and in various ASCII representations. The image and the ASCII representations for each input/output contain the same information: we just show both representations for convenience and ease of understanding. Each number corresponds to a color in the image. The correspondence is as follows: {color_to_index}."""

    abs_in_normalized_ascii_desc = """\n\nFor each shape, we indicate the non-normalized location of one cell in the grid (the first cell per shape in the prior representation) using [Absolute: LOCATION]. This is to make it easy to correspond to the prior representation and to give each shape a unique identification. We only show the absolute representation for shapes with more than 2 cells to save space."""

    if disable_absolute_in_normalized_ascii:
        abs_in_normalized_ascii_desc = ""

    # TODO: there is an extra '"' here due to a typo. Can't always fix due to cache.
    diff_rep = f"""\n\n### Color changes between the input grid and the output grid

This shows the difference between an input grid and an output grid as a list of the locations where one color changes to another. For instance, if {scheme[0]} changes to {scheme[2]} at A1 A2 B7, this would be represented as "{scheme[0]} (0) to {scheme[2]} (2): A1 A2 B7".

We will use the '...' notation as described earlier when applicable."""

    if not use_diff_rep:
        diff_rep = ""

    if connected_include_diagonals:
        explain_connected = " For this connected component representation, we use 8-connectivity (aka Moore neighborhood) where both orthogonally and diagonally adjacent pixels are considered connected. This includes pixels to the up, down, left, right, as well as the four diagonal neighbors (up-left, up-right, down-left, down-right). (Note that this differs from how (e.g.) scipy.ndimage.label treats connected components.)"
    elif use_explain_connected:
        explain_connected = " For this connected component representation, we use 4-connectivity (aka von Neumann neighborhood) where orthogonally adjacent pixels (up, down, left, right) are considered connected. (This matches scipy.ndimage.label.)"
    else:
        explain_connected = ""

    ascii_rep_desc = f"""Here are descriptions of each of the different ASCII representations we will provide:

### Color by location representation

This is a grid of elements separated by '|'. For each element, we provide the color as a number and the location (in that order). Locations are denoted like A7 or D3, where columns are denoted with A, B, C, etc., and rows are denoted with 1, 2, 3, etc. So, D3 corresponds to the cell in the 4th column and the 3rd row. Note that rows are 1-indexed.

### Location by color representation

This is a mapping from colors to the locations at which that color occurs. We use 'XR ... YR' to denote that row R is occupied from X to Y (inclusive). For instance, 'C5 ... G5' would correspond to 'C5 D5 E5 F5 G5'. We only use this '...' notation for moderately long contiguous runs of cells in a row. We don't use this notation for columns.

We also separate the list into connected components (shapes).{explain_connected} Each shape/component is separated by '|'.

### Normalized shape representation (by color)

This shows the geometry of each shape/component by "normalizing" the shape: showing the shape with the coordinates shifted such that the minimum row/column of the shape is row 1 and column A. This is useful for tasks like noticing identical shapes (in different positions with different colors).

Each shape/component is separated by '|'.{abs_in_normalized_ascii_desc}{diff_rep}

Now we're done going through the descriptions of the different ASCII representations.
""".strip()

    image_version_of_input_line = f'The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as both images and grids of numbers (ASCII). The image and the grid of numbers for each input/output contain the same information: we just show both representations for convenience. Each number corresponds to a color in the image. The correspondence is as follows: {color_to_index}.'

    maybe_diff_highlight_line = "\n\nWhen the input and output grids have identical dimensions and share the same color in more than 60% of their cells, we will display an additional version of both the input and output grids with cells that differ highlighted using a red border. This highlighting is to help you easily identify the differences between the input and output grids."

    maybe_diff_triangles_line = "\n\nWhen the input and output grids have identical dimensions and share the same color in more than 60% of their cells, we will display an additional image which shows the input color in the upper left triangle of the cell and the output color in the lower right triangle of the cell. Correspondingly, cells which are all one color (the upper triangle and lower triangle are the same color) are cells where the input and the output grids have the same color. This visualization is to help you easily identify and understand the differences between the input and output grids."

    additional_info_line_reasoning = f"""You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.

You reason in substantial detail for as long as is necessary to {'determine the transformation rule.' if not use_resolve_ambiguity else 'fully determine the transformation rule and resolve any ambiguities/uncertainties.'}"""

    no_need_conside_as_long = "\n\nYour reasoning **can be as long as necessary**! The goal of the reasoning is just to make sure you end up with a correct implementation of the transformation rule, so **there isn't any need for your reasoning to be concise**. You should do any and all reasoning that would be useful."

    if not long_as_you_want:
        no_need_conside_as_long = ""

    additional_info_line_attributes = (
        "You are creative and accomplished at solving puzzles."
    )

    additional_info_line = f"""\n\n{additional_info_line_reasoning}{no_need_conside_as_long}\n\n{additional_info_line_attributes}"""
    # print(additional_info_line)

    if just_reasoning_additional_info:
        additional_info_line = "\n\n" + additional_info_line_reasoning
        assert not just_attributes_additional_info
        assert additional_info
    elif just_attributes_additional_info:
        additional_info_line = "\n\n" + additional_info_line_attributes
        assert additional_info

    if use_many_ascii_representations:
        input_line = many_ascii_rep_and_image_version_of_input_line

        input_line += "\n\n" + ascii_rep_desc
    else:
        input_line = image_version_of_input_line

    if not use_diff_highlight:
        maybe_diff_highlight_line = ""

    if not use_diff_triangles:
        maybe_diff_triangles_line = ""

    if not additional_info:
        additional_info_line = ""

    single_correct_resolve_ambiguity = "\n\nThe transformation rule maps from each input to a single correct output, and your implementation in code must be exactly correct. Thus, you need to resolve all potential uncertainties you might have about the transformation rule. For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. Whenever there are potential ambiguities/uncertainties in your current understanding of the transformation rule, you need to resolve them before implementing the transformation in code. You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning."

    multiple_part_transformation_rule_hint = """

The transformation rule might have multiple components and might be fairly complex. It's also reasonably common that the transformation rule has one main rule (e.g., replace cells in XYZ pattern with color ABC), but has some sort of exception (e.g., don't replace cells if they have color DEF). So, you should be on the lookout for additional parts or exceptions that you might have missed so far. Consider explicitly asking yourself (in writing): \"Are there any additional parts or exceptions to the transformation rule that I might have missed?\" (Rules don't necessarily have multiple components or exceptions, but it's common enough that you should consider it.)

Here are some examples of transformation rules with multiple components or exceptions:

- There is a grey grid with black holes that have different shapes and the rule is to fill in these holes with colored cells. Further, the color to use for each hole depends on the size of the hole (in terms of the number of connected cells). 1 cell holes are filled with pink, 2 cell holes are filled with blue, and 3 cell holes are filled with red.
- The output is 3x3 while the input is 3x7. The output has red cells while the input has two "sub-grids" that are 3x3 and separated by a grey line in the middle. Each of the sub-grids has some colored cells (blue) and some black cells. The rule is to AND the two sub-grids together (i.e., take the intersection of where the two sub-grids are blue) and color the 3x3 cells in the output red if they are in the intersection and black otherwise.
- The grey rectangular outlines are filled with some color in the output. Pink, orange, and purple are used to fill in the voids in different cases. The color depends on the size of the black void inside the grey outline where it is pink if the void has 1 cell (1x1 void), orange if the gap has 4 cells, and purple if the gap was 9 cells. For each void, all of the filled-in colors are the same.
- The red shape in the input is moved. It is moved either horizontally or vertically. It is moved until moving it further would intersect with a purple shape. It is moved in the direction of the purple shape, that is, moved in whichever direction would involve it eventually intersecting with this purple shape.

These are just example rules; the actual transformation rule will be quite different. But, this should hopefully give you some sense of what transformation rules might look like.

Note that in each of these cases, you would need to find the rule by carefully examining the examples and using reasoning. You would then need to implement the transformation rule precisely, taking into account all possible cases and getting all of the details right (e.g., exactly where to place various things or exactly which color to use in each case). If the details aren't fully ironed out, you should do additional reasoning to do so before implementing the transformation in code."""

    if not use_resolve_ambiguity:
        single_correct_resolve_ambiguity = ""
    else:
        assert additional_info
        assert not just_attributes_additional_info

    if not use_multi_part_transformation_rule_hint:
        multiple_part_transformation_rule_hint = ""

    alternative_system_prompt_text = f"""You will be given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. In addition to the paired example inputs and outputs, there is also an additional input without a known output (or possibly multiple additional inputs). Your task is to determine the transformation rule and implement it in code.

{input_line}{maybe_diff_highlight_line}{maybe_diff_triangles_line}{single_correct_resolve_ambiguity}{multiple_part_transformation_rule_hint}

You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

After your reasoning, write code in triple backticks (```python and then ```). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation which works in general (for inputs which have the same properties as the example inputs and the additional input(s)).

Don't write tests in your python code, just output the `transform` function.{additional_info_line}"""

    return alternative_system_prompt_text


if __name__ == "__main__":
    print(
        get_system_prompt(
            use_diff_highlight=True,
            use_diff_triangles=True,
            additional_info=True,
            just_reasoning_additional_info=False,
            just_attributes_additional_info=False,
            use_many_ascii_representations=True,
            use_alt_color_scheme=True,
            disable_absolute_in_normalized_ascii=False,
            use_diff_rep=True,
            use_resolve_ambiguity=True,
            use_multi_part_transformation_rule_hint=True,
            use_explain_connected=True,
            connected_include_diagonals=True,
        )
    )
