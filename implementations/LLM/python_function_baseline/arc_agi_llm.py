import logging
from dotenv import load_dotenv
from litellm import completion
import subprocess
import requests
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import json
import os
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from glob import glob
from prompts import PROMPT_REASONING
from examples import example_1_same_grid_reasoning, example_1_same_grid_challenge_id
from code_parsing import parse_python_code
from copy import deepcopy
from data_loading import t, task, task_solution
from prepare_prompt import challenge_to_messages
from code_execution import run_transforms
from visualization import plot_task_complete

_ = load_dotenv()
# MODEL = "gpt-4o"
# MODEL = "claude-3-7-sonnet-20250219"
MODEL = "o3-mini"

challenge = {
    'id': t,
    'train': task['train'],
    'test': task['test']
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

# Call the model
response = completion(
    model=MODEL,
    messages=messages,
    stream=False
)


# Print the full response content
print("\n========== FULL LLM RESPONSE ==========\n")
print(response.choices[0].message.content)
print("\n======================================\n")

# Extract the code
code = parse_python_code(response.choices[0].message.content)

# Save the code to a file for inspection
code_file_path = "generated_transform_code.py"
with open(code_file_path, "w") as f:
    f.write(code)
print(f"Generated code saved to {code_file_path}")

try:
    # Execute the code
    result = run_transforms([deepcopy(test["input"]) for test in challenge["test"]], code)
    prediction = result.get_result()

    # Execute on training examples
    num_train = len(task['train'])
    for i in range(num_train):
        result = run_transforms([deepcopy(task['train'][i]['input'])], code)
        task['train'][i]['prediction'] = result.get_result()

    # Visualize results
    plot_task_complete(task, task_solution, prediction, fig_name='arc_agi_llm.png')
except Exception as e:
    print(f"Error executing code: {e}")
    print("Please check the generated_transform_code.py file for syntax errors")