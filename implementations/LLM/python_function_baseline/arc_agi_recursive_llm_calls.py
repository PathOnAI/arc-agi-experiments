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
    # Create and save the visualization
    plt.figure(figsize=(15, 10))
    plot_task_complete(task, task_solution, prediction, fig_name=fig_path)
    img_base64 = image_to_base64(fig_path)
    
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

def evaluate_solution(task, task_solution, prediction, code=None):
    """Evaluate the current solution and return metrics
    
    Args:
        task (dict): Task dictionary containing train and test examples
        task_solution (list): Solution grid for the test example
        prediction (list): Predicted output grid
        code (str, optional): The transformation code used, if available
        
    Returns:
        dict: Metrics including cell and grid accuracies
    """
    # Calculate evaluation metrics
    train_preds = [ex['prediction'] for ex in task['train']]
    train_targets = [ex['output'] for ex in task['train']]
    
    cell_accuracy = calculate_accuracy([prediction], [task_solution])
    grid_accuracy = calculate_grid_accuracy([prediction], [task_solution])
    train_cell_accuracy = calculate_accuracy(train_preds, train_targets)
    train_grid_accuracy = calculate_grid_accuracy(train_preds, train_targets)
    
    return {
        'cell_accuracy': cell_accuracy,
        'grid_accuracy': grid_accuracy,
        'train_cell_accuracy': train_cell_accuracy,
        'train_grid_accuracy': train_grid_accuracy
    }

def execute_on_examples(challenge, code):
    """Execute the transformation code on all examples
    
    Args:
        challenge (dict): Challenge dictionary containing train and test examples
        code (str): The transformation code to execute
        
    Returns:
        tuple: (task with predictions, test predictions)
    """
    # Execute on test examples
    result = run_transforms([deepcopy(test["input"]) for test in challenge["test"]], code)
    prediction = result.get_result()
    
    # Execute on training examples
    task_copy = deepcopy(challenge)
    num_train = len(task_copy['train'])
    for i in range(num_train):
        result = run_transforms([deepcopy(task_copy['train'][i]['input'])], code)
        task_copy['train'][i]['prediction'] = result.get_result()
    
    return task_copy, prediction

def iterative_llm_solution(max_iterations=5, model=MODEL):
    """Run iterative LLM solution process with specified number of iterations
    
    Args:
        max_iterations (int): Maximum number of iterations to run
        model (str): Model name to use for LLM calls
        
    Returns:
        dict: Results including best solution, metrics, and iteration history
    """
    print(f"\n===== Starting Iterative LLM Solution (max {max_iterations} iterations) =====\n")
    
    # Initialize challenge
    challenge = {
        'id': t,
        'train': task['train'],
        'test': task['test']
    }
    
    # Prepare initial messages
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
    
    # Track iteration history
    history = []
    best_solution = {
        'iteration': 0,
        'code': None,
        'metrics': {'train_grid_accuracy': 0, 'train_cell_accuracy': 0, 'grid_accuracy': 0, 'cell_accuracy': 0},
        'prediction': None
    }
    
    # Iterative solution process
    for iteration in range(1, max_iterations + 1):
        print(f"\n===== Iteration {iteration}/{max_iterations} =====\n")
        
        # Call the model
        print(f"Calling LLM for solution attempt {iteration}...")
        response = completion(
            model=model,
            messages=messages
        )
        
        # Extract and execute the code
        code = parse_python_code(response.choices[0].message.content)
        print(f"Executing solution code from iteration {iteration}...")
        
        # Execute on all examples
        task_with_preds, prediction = execute_on_examples(challenge, code)
        
        # Save visualization for this iteration
        fig_path = f'iter_{iteration}.png'
        plot_task_complete(task_with_preds, task_solution, prediction, fig_name=fig_path)
        print(f"Visualization saved to {fig_path}")
        
        # Evaluate the current solution
        metrics = evaluate_solution(task_with_preds, task_solution, prediction, code)
        print(f"Iteration {iteration} metrics:")
        print(f"  Train Grid Accuracy: {metrics['train_grid_accuracy']:.4f}")
        print(f"  Train Cell Accuracy: {metrics['train_cell_accuracy']:.4f}")
        print(f"  Test Grid Accuracy: {metrics['grid_accuracy']:.4f}")
        print(f"  Test Cell Accuracy: {metrics['cell_accuracy']:.4f}")
        
        # Get feedback from LLM
        print(f"Getting feedback for iteration {iteration}...")
        feedback = get_llm_feedback(task_with_preds, task_solution, prediction, code, fig_path=fig_path)
        
        # Save current iteration results
        current_result = {
            'iteration': iteration,
            'code': code,
            'metrics': metrics,
            'feedback': feedback,
            'prediction': prediction
        }
        history.append(current_result)
        
        # Check if this is the best solution so far - prioritizing training accuracy
        if metrics['train_grid_accuracy'] > best_solution['metrics'].get('train_grid_accuracy', 0) or \
           (metrics['train_grid_accuracy'] == best_solution['metrics'].get('train_grid_accuracy', 0) and 
            metrics['train_cell_accuracy'] > best_solution['metrics'].get('train_cell_accuracy', 0)):
            best_solution = {
                'iteration': iteration,
                'code': code,
                'metrics': metrics,
                'prediction': prediction
            }
            print(f"New best solution found at iteration {iteration}!")
        
        # Check if we've reached perfect accuracy on training examples
        if metrics['train_grid_accuracy'] == 1.0:
            print(f"Perfect training accuracy achieved at iteration {iteration}! Stopping early.")
            break
        
        # Update messages for next iteration
        messages.append({
            "role": "assistant", 
            "content": response.choices[0].message.content
        })
        
        messages.append({
            "role": "user",
            "content": f"""I received the following feedback on my solution from iteration {iteration}:

{feedback}

Based on this feedback, please provide an improved version of the transform function that better captures the transformation pattern."""
        })
    
    # Print final results
    print("\n===== Iterative LLM Solution Complete =====")
    print(f"Best solution was from iteration {best_solution['iteration']}")
    print(f"Best metrics: Train Grid Accuracy: {best_solution['metrics']['train_grid_accuracy']:.4f}, " + 
          f"Train Cell Accuracy: {best_solution['metrics']['train_cell_accuracy']:.4f}")
    print(f"Test metrics: Grid Accuracy: {best_solution['metrics']['grid_accuracy']:.4f}, " + 
          f"Cell Accuracy: {best_solution['metrics']['cell_accuracy']:.4f}")
    
    return {
        'best_solution': best_solution,
        'history': history
    }

# Main execution
if __name__ == "__main__":
    # You can change the max_iterations parameter here
    result = iterative_llm_solution(max_iterations=5)
    
    # Optionally recreate the visualization of the best solution
    best_iter = result['best_solution']['iteration']
    best_code = result['best_solution']['code']
    
    # Re-run the best solution for final visualization
    task_with_preds, prediction = execute_on_examples(
        {'id': t, 'train': task['train'], 'test': task['test']}, 
        best_code
    )
    
    # Save the best solution visualization
    plot_task_complete(task_with_preds, task_solution, prediction, fig_name='best_solution.png')
    print(f"Best solution visualization saved to best_solution.png")
    
    # You could also save the full result history for analysis
    with open('solution_history.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = []
        for item in result['history']:
            serializable_item = {k: v for k, v in item.items() if k != 'prediction'}
            history_serializable.append(serializable_item)
        
        json.dump(history_serializable, f, indent=2)