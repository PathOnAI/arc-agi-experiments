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

from code_parsing import parse_python_code

from copy import deepcopy
from data_loading import t, task, task_solution
from prepare_prompt import challenge_to_messages
from code_execution import run_challenge_transforms
from visualization import plot_task_complete



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

print(messages)
response = completion(
        model=MODEL, 
        messages=messages
    )

print(response)
code = parse_python_code(response.choices[0].message.content)
result = run_challenge_transforms(challenge, code)
print(result)
prediction = result.get_result()
print(prediction)
print(task_solution)
print(task['test'][0]['input'])
plot_task_complete(task, task_solution, prediction)
print(t)