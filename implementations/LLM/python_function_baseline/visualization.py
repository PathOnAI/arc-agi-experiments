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