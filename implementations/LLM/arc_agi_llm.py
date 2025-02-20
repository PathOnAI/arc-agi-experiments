import logging
from dotenv import load_dotenv
from litellm import completion
import subprocess
import requests
import os
import json

_ = load_dotenv()
MODEL = "gpt-4o"
MODEL = "claude-3-5-sonnet-20240620"
MODEL = "o3-mini"

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



base_path='../data/arc-prize-2024/'
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

# for i in range(5):
#     t=list(training_challenges)[i]
#     task=training_challenges[t]
#     print(f'Set #{i}, {t}')

# task = training_challenges['007bbfb7']
# print(task.keys())

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

# plt.figure(figsize=(4, 1), dpi=200)
# plt.imshow([list(range(10))], cmap=cmap, norm=norm)
# plt.xticks(list(range(10)))
# plt.yticks([])
# plt.show()

# include deep seek r1
# include openai o3-mini

# https://github.com/arcprizeorg/model_baseline/blob/main/prompt_example_o3.md

## step 1
## template for o3-mini api call and deep seek r1 api call
## step 2:
## add a data loader to transform an input idx to openai messages
## add a function to evaluate the output of the llm output to calculate the accuracy
## step 3:
## loop over the data loader, and call the llm





i = 99
t=list(training_challenges)[i]
task=training_challenges[t]
task_solution = training_solutions[t][0]





# prompt = "Find the common rule that maps an input grid to an output grid, given the examples below. Example 1: Input: 0 0 0 5 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 0 0 Output: 1 0 0 0 0 0 5 5 0 0 0 1 0 0 0 0 5 5 0 0 0 0 5 5 0 0 0 0 1 0 0 0 5 5 0 0 0 0 0 1 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 5 5 0 0 1 0 0 0 0 0 5 5 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 Example 2: Input: 2 0 0 0 Output: 2 2 0 0 2 2 0 0 0 0 1 0 0 0 0 1 Example 3: Input: 0 0 0 0 0 3 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Output: 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 3 3 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Your final answer should just be the text output grid itself. Input: 0 4 0 0 0 0 4 0 0"
# print(prompt)
# print(task)
# print(task_solution)

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

print(prompt)



# Initialize messages
messages = [{
    "role": "system", 
    "content": prompt
}]

# messages.append({"role": "user", "content": 'calculate 1+1'})
response = completion(
        model=MODEL, 
        messages=messages
    )

print(response)