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
import io
import base64
from PIL import Image
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
MODEL = "gpt-4o"  # Using GPT-4o for more advanced visualization analysis capabilities
# MODEL = "claude-3-5-sonnet-20240620"
# MODEL = "o3-mini"

def calculate_accuracy(prediction, target):
    """Calculate cell-wise accuracy between prediction and target grids"""
    correct = 0
    total = 0
    
    for pred_grid, target_grid in zip(prediction, target):
        pred_flat = [item for sublist in pred_grid for item in sublist]
        target_flat = [item for sublist in target_grid for item in sublist]
        
        for p, t in zip(pred_flat, target_flat):
            if p == t:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0

def calculate_grid_accuracy(predictions, targets):
    """Calculate grid-wise accuracy (whole grid must match)"""
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if np.array_equal(pred, target):
            correct += 1
    
    return correct / total if total > 0 else 0

def image_to_base64(image_path):
    """
    Convert an image file to a base64 encoded string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    import base64
    
    # Read the image file
    with open(image_path, "rb") as image_file:
        # Encode the binary data to base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return encoded_string

def get_llm_feedback(task, task_solution, prediction, code, fig_path=None):
    """Get feedback from LLM on the predicted solution
    
    Args:
        task (dict): Task dictionary containing train and test examples
        task_solution (list): Solution grid for the test example
        prediction (list): Predicted output grid
        code (str): The transformation code to evaluate
        fig_path (str, optional): Path to save the visualization figure. If None, a default name is used.
    """
    # # Calculate evaluation metrics
    # train_preds = [ex['prediction'][0] for ex in task['train']]
    # train_targets = [ex['output'] for ex in task['train']]
    
    # cell_accuracy = calculate_accuracy(prediction, task_solution)
    # grid_accuracy = calculate_grid_accuracy(prediction, task_solution)
    # train_cell_accuracy = calculate_accuracy(train_preds, train_targets)
    # train_grid_accuracy = calculate_grid_accuracy(train_preds, train_targets)
    
    # Create and save the visualization
    plt.figure(figsize=(15, 10))
    feedback_fig_name = fig_path if fig_path is not None else 'feedback_visualization.png'
    plot_task_complete(task, task_solution, prediction, fig_name=feedback_fig_name)
    img_base64 = image_to_base64(feedback_fig_name)
    
    # Prepare prompt for feedback with image included
    feedback_prompt = [
        {
            "role": "system",
            "content": """You are an AI code evaluator specialized in analyzing grid-based transformation algorithms. 
            Review the solution and provide detailed, constructive feedback on:
            1. Pattern recognition - if the code correctly identified transformation rules
            2. Edge cases - any specific examples where the solution failed
            3. Code quality - suggestions for improvement
            
            Be specific with your feedback, referencing the visual results shown in the image."""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""I've attempted to solve a grid transformation problem. Here's my solution:

    ## The transformation code I wrote:
    ```python
    {code}
    ```

    Below is a visualization showing:
    - Top rows: Training examples with input, expected output, and my prediction
    - Bottom row: Test examples with input, ground truth, and my prediction

    Please analyze the image and provide detailed feedback on my solution. What patterns did I capture correctly or miss?
    How can I improve my code? Are there specific examples where my solution failed?"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        }
    ]

    # Get feedback from LLM
    feedback_response = completion(
        model=MODEL,
        messages=feedback_prompt
    )
    
    return feedback_response.choices[0].message.content

# Main execution
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

# # Print message length for debugging
# print(f"Prompt token length estimate: {len(str(messages)) / 4}")

# Call the model
response = completion(
    model=MODEL,
    messages=messages
)

# Print the full response content
print("\n========== FULL LLM RESPONSE ==========\n")
print(response.choices[0].message.content)
print("\n======================================\n")

# Extract and execute the code
code = parse_python_code(response.choices[0].message.content)
result = run_transforms([deepcopy(test["input"]) for test in challenge["test"]], code)
prediction = result.get_result()

# Execute on training examples
num_train = len(task['train'])
for i in range(num_train):
    result = run_transforms([deepcopy(task['train'][i]['input'])], code)
    task['train'][i]['prediction'] = result.get_result()

# Visualize results (this will be visible to the user)
plot_task_complete(task, task_solution, prediction, fig_name='iter_1.png')

# Get feedback from LLM
print("\n========== GENERATING FEEDBACK ==========\n")
feedback = get_llm_feedback(task, task_solution, prediction, code)

# # Print metrics and feedback
# print("\n========== SOLUTION METRICS ==========")
# print(f"Test Cell Accuracy: {metrics['cell_accuracy']:.4f}")
# print(f"Test Grid Accuracy: {metrics['grid_accuracy']:.4f}")
# print(f"Training Cell Accuracy: {metrics['train_cell_accuracy']:.4f}")
# print(f"Training Grid Accuracy: {metrics['train_grid_accuracy']:.4f}")

print("\n========== LLM FEEDBACK ==========\n")
print(feedback)


# Append the original solution and feedback to messages for the next iteration
messages.append({
    "role": "assistant", 
    "content": "the generated code is {}".format(code)
})

messages.append({
    "role": "user",
    "content": f"""I received the following feedback on my solution:

{feedback}

Based on this feedback, please provide an improved version of the transform function that better captures the transformation pattern."""
})

# Get improved solution from LLM
response = completion(
    model=MODEL,
    messages=messages
)


# Print the full response content
print("\n========== FULL LLM RESPONSE ==========\n")
print(response.choices[0].message.content)
print("\n======================================\n")

# Extract and execute the code
code = parse_python_code(response.choices[0].message.content)
result = run_transforms([deepcopy(test["input"]) for test in challenge["test"]], code)
prediction = result.get_result()

# Execute on training examples
num_train = len(task['train'])
for i in range(num_train):
    result = run_transforms([deepcopy(task['train'][i]['input'])], code)
    task['train'][i]['prediction'] = result.get_result()

# Visualize results (this will be visible to the user)
plot_task_complete(task, task_solution, prediction, fig_name='iter_2.png')
