import logging
from dotenv import load_dotenv
from litellm import completion
import subprocess
import requests
import os
import json

_ = load_dotenv()
MODEL = "gpt-4o"
# MODEL = "claude-3-5-sonnet-20240620"
# MODEL = "o3-mini"

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from   matplotlib import colors
import seaborn as sns

import json
import os
from pathlib import Path

from subprocess import Popen, PIPE, STDOUT
from glob import glob
from prompts import PROMPT_REASONING
from examples import example_1_same_grid_reasoning,  example_1_same_grid_challenge_id



base_path='../../data/arc-prize-2024/'
# Loading JSON data
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Reading files
training_challenges =  load_json(base_path +'arc-agi_training_challenges.json')
training_solutions =   load_json(base_path +'arc-agi_training_solutions.json')

evaluation_challenges =load_json(base_path +'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_json(base_path +'arc-agi_evaluation_solutions.json')

test_challenges =  load_json(base_path +'arc-agi_test_challenges.json')

print(f'Number of training challenges = {len(training_challenges)}')
print(f'Number of training solutions = {len(training_solutions)}')
print(f'Number of evaluation challenges = {len(evaluation_challenges)}')
print(f'Number of evaluation solutions = {len(evaluation_solutions)}')
print(f'Number of test challenges = {len(test_challenges)}')

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

t = '447fd412'
task=training_challenges[t]
task_solution = training_solutions[t][0]

# Format training examples
train_examples = ""
for i, example in enumerate(task['train'], 1):
    # Convert input and output grids to strings
    input_str = ' '.join(str(num) for row in example['input'] for num in row)
    output_str = ' '.join(str(num) for row in example['output'] for num in row)
    train_examples += f"Example {i}: Input: {input_str} Output: {output_str} "



def format_grid(grid, width=10):
    """Format a flat list of numbers into rows with width"""
    rows = [' '.join(map(str, grid[i:i+width])) for i in range(0, len(grid), width)]
    return '\n'.join(rows)

# Format training examples
train_examples = ""
for i, example in enumerate(task['train'], 1):
    input_str = format_grid([num for row in example['input'] for num in row])
    output_str = format_grid([num for row in example['output'] for num in row], width=2)
    train_examples += f"Example {i}:\nInput:\n{input_str}\nOutput:\n{output_str}\n\n"

# Format test input
test_input = format_grid([num for row in task['test'][0]['input'] for num in row])

prompt = """Find the common rule that maps an input grid to an output grid, given the examples below.\n\n{}\nBelow is a test input grid. Predict the corresponding output grid by applying the rule you found. Your final answer should just be the text output grid itself.\n\nInput:\n{}""".format(train_examples, test_input)


def grid_to_ascii(grid, separator='|', spreadsheet_ascii=False):
    """Convert a grid (2D list) to ASCII representation."""
    result = []
    for row in grid:
        row_str = separator.join(str(cell) for cell in row)
        result.append(row_str)
    return '\n'.join(result)

def array_to_str(grid):
    """Convert a grid to array string representation."""
    return str(grid)

def does_grid_change_shape(challenge):
    """Check if any example in the challenge changes grid shape from input to output."""
    for train_example in challenge['train']:
        input_shape = (len(train_example['input']), len(train_example['input'][0]))
        output_shape = (len(train_example['output']), len(train_example['output'][0]))
        if input_shape != output_shape:
            return True
    return False

def grid_diffs_to_ascii(grid_input, grid_output, separator='|'):
    """Show differences between input and output grids in ASCII."""
    result = []
    for i in range(len(grid_input)):
        row = []
        for j in range(len(grid_input[0])):
            if grid_input[i][j] != grid_output[i][j]:
                row.append(f"{grid_input[i][j]}->{grid_output[i][j]}")
            else:
                row.append(str(grid_input[i][j]))
        result.append(separator.join(row))
    return '\n'.join(result)

def content_blocks_from_matrix(matrix, label, include_image=False, use_ascii=True, use_array=True):
    """Generate content blocks for a matrix with specified formatting options."""
    grid = matrix
    x, y = len(grid), len(grid[0])
    messages = [
        {"type": "text", "text": label},
        {"type": "text", "text": f"Shape: {x} by {y}\n\n"},
    ]
    
    if use_ascii:
        messages.append({
            "type": "text",
            "text": f"ASCII representation:\n\n{grid_to_ascii(grid=grid, separator='|')}\n\n",
        })
    if use_array:
        messages.append({"type": "text", "text": array_to_str(grid=matrix)})
    return messages

def content_from_challenge(challenge, include_diffs=True, include_image=False, use_ascii=True, use_array=True):
    """Generate content blocks for an entire challenge."""
    content = []
    for i, train in enumerate(challenge['train']):
        example_number = i + 1
        # add input blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train['input'],
                label=f"# Example {example_number}\n\n## Input {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        # add output blocks
        content.extend(
            content_blocks_from_matrix(
                matrix=train['output'],
                label=f"## Output {example_number}\n\n",
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        )
        if not does_grid_change_shape(challenge) and include_diffs:
            content.append({
                "type": "text",
                "text": f"## Color changes between the Input and Output ASCII representation:\n\n"
                    f"{grid_diffs_to_ascii(grid_input=train['input'], grid_output=train['output'], separator='|')}\n\n",
            })

    content.extend(
        content_blocks_from_matrix(
            matrix=challenge['test'][0]['input'],
            label="# Additional input\n\n",
            include_image=include_image,
            use_ascii=use_ascii,
            use_array=use_array,
        )
    )
    return content

def challenge_to_messages(
    challenge,
    prompt_text,
    add_examples=True,
    use_cache_control=True,
    include_diffs=True,
    include_image=False,
    use_ascii=True,
    use_array=True
):
    """
    Convert a challenge to a list of messages suitable for an LLM conversation.
    
    Args:
        challenge (dict): Challenge dictionary with 'id', 'train', and 'test' keys
        prompt_text (str): The system prompt text to use
        add_examples (bool): Whether to add example challenges
        use_cache_control (bool): Whether to add cache control metadata
        include_diffs (bool): Whether to include difference visualizations
        include_image (bool): Whether to include image representations
        use_ascii (bool): Whether to include ASCII representations
        use_array (bool): Whether to include array string representations
    
    Returns:
        list: List of message dictionaries for the conversation
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt_text}]}
    ]

    if add_examples:
        # Here you would add your example challenges
        # For brevity, example challenge insertion is omitted
        example_1_grid_same_prompt = content_from_challenge(
                challenge=training_challenges[example_1_same_grid_challenge_id],
                include_diffs=include_diffs,
                include_image=include_image,
                use_ascii=use_ascii,
                use_array=use_array,
            )
        print(example_1_grid_same_prompt)
        print(example_1_same_grid_reasoning)
        messages.extend([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Great work! Now I will give you another puzzle to solve just like that one.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Great, please give me the next puzzle.",
                    }
                ],
            },
        ])

    if use_cache_control:
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    content = content_from_challenge(
        challenge=challenge,
        include_diffs=include_diffs,
        include_image=include_image,
        use_ascii=use_ascii,
        use_array=use_array,
    )
    
    if use_cache_control:
        content[-1]["cache_control"] = {"type": "ephemeral"}
    
    messages.append({"role": "user", "content": content})
    return messages

challenge = {
    'id': t,
    'train': task['train'],  # already in the correct format of [{'input': [...], 'output': [...]}]
    'test': task['test']     # already in the correct format
}

messages = challenge_to_messages(
    challenge=challenge,
    prompt_text=PROMPT_REASONING,
    add_examples=True,
    use_cache_control=True,
    include_diffs=True,
    include_image=False,
    use_ascii=True,
    use_array=True
)

print(messages)


# # Initialize messages
# messages = [{
#     "role": "system", 
#     "content": prompt
# }]

# # messages.append({"role": "user", "content": 'calculate 1+1'})
response = completion(
        model=MODEL, 
        messages=messages
    )

print(response)


## extract the code from the response

## apply the code to the test input

## show the model prediction

## show the test output

import re
from typing import Optional

class CodeBlockParser:
    """Parser for extracting and cleaning Python code blocks from text."""
    
    DEFAULT_CODE = """
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    raise NotImplementedError()
""".strip()

    def __init__(self, tab_size: int = 4):
        self.tab_size = tab_size
        
    def clean_whitespace(self, code: str) -> str:
        """Normalize whitespace by converting tabs to spaces."""
        return code.replace("\t", " " * self.tab_size)

    def extract_transform_function(self, text: str) -> str:
        """
        Extract the transform function from text containing Python code blocks.
        Returns the DEFAULT_CODE if no valid code block is found.
        """
        # Handle case with no code blocks
        if "```python" not in text:
            content = text.partition("</reasoning>")[2]
            return self.clean_whitespace(content) if content else self.DEFAULT_CODE

        # Find all Python code blocks
        code_blocks = text.split("```python")
        
        # If multiple blocks exist, find the one with transform function
        if len(code_blocks) > 1:
            for block in reversed(code_blocks):
                if "def transform(" in block:
                    return self._parse_code_block("```python" + block)
        
        # Handle single code block
        return self._parse_code_block(text)

    def _parse_code_block(self, text: str) -> str:
        """Parse a single code block using different patterns."""
        patterns = [
            r"```python\n(.*)\n```",  # Standard pattern
            r"```python\n(.*)\n`",    # Alternative ending
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                return self.clean_whitespace(match.group(1))
        
        # Fallback: try to extract everything after ```python
        remaining_code = text.partition("```python")[2]
        return self.clean_whitespace(remaining_code) if remaining_code else self.DEFAULT_CODE

# Example usage:
def parse_python_code(text: str) -> str:
    """
    Main interface function for parsing Python code blocks.
    Returns cleaned code with normalized whitespace.
    """
    parser = CodeBlockParser()
    return parser.extract_transform_function(text)

from copy import deepcopy

code = parse_python_code(response.choices[0].message.content)


import json
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from typing import List, Optional, Dict, Any, TypeVar, Union

# Type alias for grid
GRID = List[List[int]]

class PythonException(Exception):
    """Custom exception for Python transform execution errors."""
    pass

class PythonResult:
    def __init__(
        self,
        stdout: str,
        stderr: str,
        return_code: int,
        timed_out: bool,
        latency_ms: float,
        transform_results: Optional[List[GRID]]
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.timed_out = timed_out
        self.latency_ms = latency_ms
        self._transform_results = transform_results
        
    @property
    def transform_results(self) -> Optional[List[GRID]]:
        """Get the transformed results as Python objects."""
        if not self._transform_results:
            # Try to parse from stdout if not already parsed
            try:
                for line in self.stdout.splitlines():
                    if line.startswith("TRANSFORM_RESULT:"):
                        results_str = line.replace("TRANSFORM_RESULT:", "", 1)
                        self._transform_results = json.loads(results_str)
                        break
            except (json.JSONDecodeError, AttributeError):
                pass
        return self._transform_results
        
    def get_result(self, index: int = 0) -> Optional[GRID]:
        """Get a specific result grid by index."""
        results = self.transform_results
        if results and 0 <= index < len(results):
            return results[index]
        return None

class PythonTransformExecutor:
    """Executes Python transform functions in a subprocess with timeout."""

    TRANSFORM_RESULT_PREFIX = "TRANSFORM_RESULT:"

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def _create_wrapped_code(self, transform_code: str, grid_lists: List[GRID]) -> str:
        """Create the complete Python code to execute with necessary imports and error handling."""
        return f"""
import json
import sys
import numpy as np
import scipy
from typing import List, Tuple, Set, Union, Optional

# Original transform function
{transform_code}

def to_python_array(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return to_python_array(obj.tolist())
    elif isinstance(obj, list):
        return [to_python_array(item) for item in obj]
    return obj

def main():
    try:
        grid_lists = {json.dumps(grid_lists)}
        results = []
        
        for grid_list in grid_lists:
            result = transform(grid_list)
            result = to_python_array(result)
            
            if not isinstance(result, list) or not all(isinstance(row, list) for row in result):
                print("Error: transform must return List[List[int]]", file=sys.stderr)
                sys.exit(1)
            
            results.append(result)
        
        print("{self.TRANSFORM_RESULT_PREFIX}" + json.dumps(results))
        
    except Exception as e:
        print(f"Error executing transform: {{str(e)}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

    def execute_transform(
        self,
        code: str,
        grid_lists: List[GRID],
        raise_exception: bool = True
    ) -> PythonResult:
        """
        Execute a Python transform function with the provided grid lists.
        
        Args:
            code: Python code containing the transform function
            grid_lists: List of input grids to transform
            raise_exception: Whether to raise an exception on error
            
        Returns:
            PythonResult containing execution results and transformed grids
        """
        start_time = time.time()
        wrapped_code = self._create_wrapped_code(code, grid_lists)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped_code)
            temp_file = f.name

        try:
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            transform_results = None
            timed_out = False

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                return_code = process.returncode

                if return_code == 0 and stdout:
                    for line in stdout.splitlines():
                        if line.startswith(self.TRANSFORM_RESULT_PREFIX):
                            try:
                                transform_results = json.loads(
                                    line.replace(self.TRANSFORM_RESULT_PREFIX, "", 1)
                                )
                            except json.JSONDecodeError:
                                stderr = "Error: Could not parse transform result"
                                return_code = 1

            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                stderr = f"Execution timed out after {self.timeout} seconds"
                return_code = -1
                timed_out = True

            latency_ms = (time.time() - start_time) * 1000

            if not transform_results and raise_exception:
                raise PythonException(stderr)

            return PythonResult(
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
                timed_out=timed_out,
                latency_ms=latency_ms,
                transform_results=transform_results
            )

        finally:
            os.unlink(temp_file)

def run_python_transform_sync(
    code: str,
    grid_lists: List[GRID],
    timeout: int = 5,
    raise_exception: bool = True
) -> PythonResult:
    """Convenience function to execute a Python transform."""
    executor = PythonTransformExecutor(timeout=timeout)
    return executor.execute_transform(code, grid_lists, raise_exception)

def run_challenge_transforms(challenge, code: str) -> PythonResult:
    return run_python_transform_sync(
        code=code,
        grid_lists=[deepcopy(test["input"]) for test in challenge["test"]],
        timeout=5,
        raise_exception=True
    )


result = run_challenge_transforms(challenge, code)
print(result)
prediction = result.get_result()
print(prediction)
print(task_solution)
print(task['test'][0]['input'])


import matplotlib.pyplot as plt
import numpy as np

def create_color_map():
    """Create color map for visualization similar to ARC interface."""
    colors = ['#FFFFFF', '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', 
              '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF']
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(range(len(colors)+1)), len(colors))
    return cmap, norm

def plot_grid(ax, grid, title):
    """Plot a single grid with proper formatting."""
    cmap, norm = create_color_map()
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_xticks([x-0.5 for x in range(1 + len(grid[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(grid))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def plot_task_complete(task, task_solution, prediction=None):
    """
    Plot a complete task with training examples, test input, solution, and prediction.
    
    Args:
        task (dict): Task dictionary containing train and test examples
        prediction (list): Optional predicted output grid
    """
    num_train = len(task['train'])
    
    # Calculate number of rows and columns needed
    num_rows = num_train + 1  # Training examples + test
    num_cols = 3  # Input, output, and solution/prediction
    
    # Create figure with smaller dimensions
    fig = plt.figure(figsize=(12, 2.5 * num_rows))
    
    # Plot training examples
    for i in range(num_train):
        # Input subplot - first column
        ax = plt.subplot2grid((num_rows, num_cols), (i, 0))
        plot_grid(ax, task['train'][i]['input'], f'Train {i+1} Input')
        
        # Output subplot - second column
        ax = plt.subplot2grid((num_rows, num_cols), (i, 1))
        plot_grid(ax, task['train'][i]['output'], f'Train {i+1} Output')
        
        # Third column remains empty for training examples
        ax = plt.subplot2grid((num_rows, num_cols), (i, 2))
        ax.remove()
    
    # Plot test row
    # Test input - first column
    ax = plt.subplot2grid((num_rows, num_cols), (num_rows-1, 0))
    plot_grid(ax, task['test'][0]['input'], 'Test Input')
    
    # Test solution - second column
    ax = plt.subplot2grid((num_rows, num_cols), (num_rows-1, 1))
    plot_grid(ax, task_solution, 'Test Solution')
    
    # Prediction - third column if provided
    if prediction is not None:
        ax = plt.subplot2grid((num_rows, num_cols), (num_rows-1, 2))
        plot_grid(ax, prediction, 'Prediction')
    else:
        ax = plt.subplot2grid((num_rows, num_cols), (num_rows-1, 2))
        ax.remove()
    
    # Adjust layout and style
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    # Style the figure
    fig.patch.set_facecolor('#dddddd')
    fig.patch.set_linewidth(3)  # Reduced from 5 to 3
    fig.patch.set_edgecolor('black')
    
    plt.show()

plot_task_complete(task, task_solution, prediction)

print(t)