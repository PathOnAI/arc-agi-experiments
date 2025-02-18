PROMPT_FROM_RYAN_PYTHON = """
You will be given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. In addition to the paired example inputs and outputs, there is also an additional input without a known output (or possibly multiple additional inputs). Your task is to determine the transformation rule and implement it in code.

The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as images and in various ASCII representations. The image and the ASCII representations for each input/output contain the same information: we just show both representations for convenience and ease of understanding. Each number corresponds to a color in the image. The correspondence is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.

Here are descriptions of each of the different ASCII representations we will provide:

### Matrix representation: This will be a 2d matrix of numbers corresponding to the colors.

The transformation rule maps from each input to a single correct output, and your implementation in code must be exactly correct. Thus, you need to resolve all potential uncertainties you might have about the transformation rule. For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. Whenever there are potential ambiguities/uncertainties in your current understanding of the transformation rule, you need to resolve them before implementing the transformation in code. You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning.

The transformation rule might have multiple components and might be fairly complex. It's also reasonably common that the transformation rule has one main rule (e.g., replace cells in XYZ pattern with color X), but has some sort of exception (e.g., don't replace cells if they have color Y). So, you should be on the lookout for additional parts or exceptions that you might have missed so far. Consider explicitly asking yourself (in writing): "Are there any additional parts or exceptions to the transformation rule that I might have missed?" (Rules don't necessarily have multiple components or exceptions, but it's common enough that you should consider it.)

Here are some examples of transformation rules with multiple components or exceptions:

- There is a grey grid with black holes that have different shapes and the rule is to fill in these holes with colored cells. Further, the color to use for each hole depends on the size of the hole (in terms of the number of connected cells). 1 cell holes are filled with pink, 2 cell holes are filled with blue, and 3 cell holes are filled with red.
- The output is 3x3 while the input is 3x7. The output has red cells while the input has two "sub-grids" that are 3x3 and separated by a grey line in the middle. Each of the sub-grids has some colored cells (blue) and some black cells. The rule is to AND the two sub-grids together (i.e., take the intersection of where the two sub-grids are blue) and color the 3x3 cells in the output red if they are in the intersection and black otherwise.
- The grey rectangular outlines are filled with some color in the output. Pink, orange, and purple are used to fill in the voids in different cases. The color depends on the size of the black void inside the grey outline where it is pink if the void has 1 cell (1x1 void), orange if the gap has 4 cells, and purple if the gap was 9 cells. For each void, all of the filled-in colors are the same.
- The red shape in the input is moved. It is moved either horizontally or vertically. It is moved until moving it further would intersect with a purple shape. It is moved in the direction of the purple shape, that is, moved in whichever direction would involve it eventually intersecting with this purple shape.

These are just example rules; the actual transformation rule will be quite different. But, this should hopefully give you some sense of what transformation rules might look like.

Note that in each of these cases, you would need to find the rule by carefully examining the examples and using reasoning. You would then need to implement the transformation rule precisely, taking into account all possible cases and getting all of the details right (e.g., exactly where to place various things or exactly which color to use in each case). If the details aren't fully ironed out, you should do additional reasoning to do so before implementing the transformation in code.

You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

After your reasoning, write code in triple backticks (```python and then ```). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation which works in general (for inputs which have the same properties as the example inputs and the additional input(s)).

Don't write tests in your python code, just output the `transform` function.

You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.

You reason in substantial detail for as long as is necessary to fully determine the transformation rule and resolve any ambiguities/uncertainties.

You are creative and accomplished at solving puzzles.
"""

PYTHON_PART = """
After your reasoning, write code in triple backticks (```python and then ```).
You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).
Your python code should not use libraries outside of the standard python libraries besides numpy.You can create helper functions.
You should make sure that you implement a version of the transformation which works in general (for inputs which have the same properties as the example inputs and the additional input(s)).
Don't write tests in your python code, just output the `transform` function.
""".strip()

DIRECT_GRID_NO_THOUGHTS_PART = """
After your reasoning, just give valid json list of lists response back, nothing else. Do not explain your thoughts.
""".strip()

# helpful for the debugging step
DIRECT_GRID_WITH_THOUGHTS_PART = """
After your reasoning, just give valid json list of lists response back, nothing else.
""".strip()

REASONING_PART = """
You will be given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. In addition to the paired example inputs and outputs, there is also an additional input without a known output (or possibly multiple additional inputs). Your task is to determine the transformation rule and implement it in code.

The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as both images and grids of numbers (ASCII). You may also get an ASCII representation of the change for each cell comparing the input and output. For example 0 -> 1 means the value switched from 0 to 1 in that cell. `--` means the value in that cell did not change. The image and the grid of numbers for each input/output contain the same information: we just show both representations for convenience. Each number corresponds to a color in the image. The correspondence is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.

The transformation rule maps from each input to a single correct output, and your implementation in code must be exactly correct. Thus, you need to resolve all potential uncertainties you might have about the transformation rule. For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. Whenever there are potential ambiguities or uncertainties in your current understanding of the transformation rule, you need to resolve them before implementing the transformation in code. You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning.

The transformation rule might have multiple components and might be fairly complex. It's also reasonably common that the transformation rule has one main rule (e.g., replace cells in XYZ pattern with color ABC), but has some sort of exception (e.g., don't replace cells if they have color DEF). So, you should be on the lookout for additional parts or exceptions that you might have missed so far. Consider explicitly asking yourself (in writing): "Are there any additional parts or exceptions to the transformation rule that I might have missed?" (Rules don't necessarily have multiple components or exceptions, but it's common enough that you should consider it.)

Here are some examples of transformation rules with multiple components or exceptions:

- There is a grey grid with black holes that have different shapes and the rule is to fill in these holes with colored cells. Further, the color to use for each hole depends on the size of the hole (in terms of the number of connected cells). 1 cell holes are filled with pink, 2 cell holes are filled with blue, and 3 cell holes are filled with red.
- The output is 3x3 while the input is 3x7. The output has red cells while the input has two "sub-grids" that are 3x3 and separated by a grey line in the middle. Each of the sub-grids has some colored cells (blue) and some black cells. The rule is to AND the two sub-grids together (i.e., take the intersection of where the two sub-grids are blue) and color the 3x3 cells in the output red if they are in the intersection and black otherwise.
- The grey rectangular outlines are filled with some color in the output. Pink, orange, and purple are used to fill in the voids in different cases. The color depends on the size of the black void inside the grey outline where it is pink if the void has 1 cell (1x1 void), orange if the gap has 4 cells, and purple if the gap was 9 cells. For each void, all of the filled-in colors are the same.
- The red shape in the input is moved. It is moved either horizontally or vertically. It is moved until moving it further would intersect with a purple shape. It is moved in the direction of the purple shape, that is, moved in whichever direction would involve it eventually intersecting with this purple shape.

These are just example rules; the actual transformation rule will be quite different. But, this should hopefully give you some sense of what transformation rules might look like.

Note that in each of these cases, you would need to find the rule by carefully examining the examples and using reasoning. You would then need to implement the transformation rule precisely, taking into account all possible cases and getting all of the details right (e.g., exactly where to place various things or exactly which color to use in each case). If the details aren't fully ironed out, you should do additional reasoning to do so before implementing the transformation in code.

You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.

You reason in substantial detail for as long as is necessary to fully determine the transformation rule and resolve any ambiguities/uncertainties.
""".strip()


COT_PART = """
You will be given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. In addition to the paired example inputs and outputs, there is also an additional input without a known output (or possibly multiple additional inputs). Your task is to determine the transformation rule and implement it in code.

The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as images and in various ASCII representations. The image and the ASCII representations for each input/output contain the same information: we just show both representations for convenience and ease of understanding. Each number corresponds to a color in the image. The correspondence is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.

Here are descriptions of each of the different ASCII representations we will provide:

### Matrix representation: This will be a 2d matrix of numbers corresponding to the colors.

The transformation rule maps from each input to a single correct output, and your implementation in code must be exactly correct. Thus, you need to resolve all potential uncertainties you might have about the transformation rule. For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. Whenever there are potential ambiguities/uncertainties in your current understanding of the transformation rule, you need to resolve them before implementing the transformation in code. You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning.

The transformation rule might have multiple components and might be fairly complex. It's also reasonably common that the transformation rule has one main rule (e.g., replace cells in XYZ pattern with color X), but has some sort of exception (e.g., don't replace cells if they have color Y). So, you should be on the lookout for additional parts or exceptions that you might have missed so far. Consider explicitly asking yourself (in writing): "Are there any additional parts or exceptions to the transformation rule that I might have missed?" (Rules don't necessarily have multiple components or exceptions, but it's common enough that you should consider it.)

Here are some examples of transformation rules with multiple components or exceptions:

- There is a grey grid with black holes that have different shapes and the rule is to fill in these holes with colored cells. Further, the color to use for each hole depends on the size of the hole (in terms of the number of connected cells). 1 cell holes are filled with pink, 2 cell holes are filled with blue, and 3 cell holes are filled with red.
- The output is 3x3 while the input is 3x7. The output has red cells while the input has two "sub-grids" that are 3x3 and separated by a grey line in the middle. Each of the sub-grids has some colored cells (blue) and some black cells. The rule is to AND the two sub-grids together (i.e., take the intersection of where the two sub-grids are blue) and color the 3x3 cells in the output red if they are in the intersection and black otherwise.
- The grey rectangular outlines are filled with some color in the output. Pink, orange, and purple are used to fill in the voids in different cases. The color depends on the size of the black void inside the grey outline where it is pink if the void has 1 cell (1x1 void), orange if the gap has 4 cells, and purple if the gap was 9 cells. For each void, all of the filled-in colors are the same.
- The red shape in the input is moved. It is moved either horizontally or vertically. It is moved until moving it further would intersect with a purple shape. It is moved in the direction of the purple shape, that is, moved in whichever direction would involve it eventually intersecting with this purple shape.

These are just example rules; the actual transformation rule will be quite different. But, this should hopefully give you some sense of what transformation rules might look like.

Note that in each of these cases, you would need to find the rule by carefully examining the examples and using reasoning. You would then need to implement the transformation rule precisely, taking into account all possible cases and getting all of the details right (e.g., exactly where to place various things or exactly which color to use in each case). If the details aren't fully ironed out, you should do additional reasoning to do so before implementing the transformation in code.

<chain of thought instruction>
Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches.
Break down the solution into clear steps within <step> tags. Start with a 20-step budget, requesting more for complex problems if needed.
Use <count> tags after each step to show the remaining budget. Stop when reaching 0.
Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
Regularly evaluate progress using <reflection> tags. Be critical and honest about your reasoning process.
Assign a quality score between 0.0 and 1.0 using <reward> tags after each reflection. Use this to guide your approach:

0.8+: Continue current approach
0.5-0.7: Consider minor adjustments
Below 0.5: Seriously consider backtracking and trying a different approach

If unsure or if reward score is low, backtrack and try a different approach, explaining your decision within <thinking> tags.
For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs.
Explore multiple solutions individually if possible, comparing approaches in reflections.
Use thoughts as a scratchpad, writing out all calculations and reasoning explicitly.
</chain of thought instruction>
""".strip()

SMALL_PART = """
You will be given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. In addition to the paired example inputs and outputs, there is also an additional input without a known output (or possibly multiple additional inputs). Your task is to determine the transformation rule and implement it in code.

The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as both images and grids of numbers (ASCII). The image and the grid of numbers for each input/output contain the same information: we just show both representations for convenience. Each number corresponds to a color in the image. The correspondence is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.

The transformation rule maps from each input to a single correct output, and your implementation in code must be exactly correct. Thus, you need to resolve all potential uncertainties you might have about the transformation rule. For instance, if the examples always involve some particular color being changed to another color in the output, but which color it is changed to varies between different examples, then you need to figure out what determines the correct output color. As another example, if some shape(s) or cells in the input are relocated or recolored, you need to determine which exact shapes should be relocated/recolored in the output and where they should be moved or what their color in the output should be. Whenever there are potential ambiguities/uncertainties in your current understanding of the transformation rule, you need to resolve them before implementing the transformation in code. You should resolve ambiguities and uncertainties by carefully analyzing the examples and using step by step reasoning.

You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.

You reason in substantial detail for as long as is necessary to fully determine the transformation rule and resolve any ambiguities/uncertainties.
""".strip()

COT_PROMPT_PYTHON = f"""
{COT_PART}

{PYTHON_PART}
""".strip()

COT_PROMPT_GRID = f"""
{COT_PART}

{DIRECT_GRID_WITH_THOUGHTS_PART}
""".strip()

REASONING_PROMPT_PYTHON = f"""
{REASONING_PART}

{PYTHON_PART}
""".strip()

REASONING_PROMPT_GRID = f"""
{REASONING_PART}

{DIRECT_GRID_WITH_THOUGHTS_PART}
""".strip()

DIRECT_GRID_NO_THOUGHTS_PROMPT = """
You are a bot that is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
Identify the pattern, then apply that pattern to the test input to give a final output.
Just give valid json list of lists response back, nothing else. Do not explain your thoughts.
""".strip()
