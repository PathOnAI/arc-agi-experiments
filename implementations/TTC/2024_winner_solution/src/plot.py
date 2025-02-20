import math

import matplotlib.pyplot as plt
from devtools import pformat
from matplotlib import colors

from src.models import Attempt
from src.prompts.colors import color_map


def plot_grid(
    input_matrix: list[list[int]],
    attempt_matrix: list[list[int]],
    solution_matrix: list[list[int]],
    title: str,
    axs: list[plt.Axes] = None,
    show: bool = True,
) -> plt.Figure:
    cmap = colors.ListedColormap([v.hex for v in color_map.values()])
    norm = colors.Normalize(vmin=0, vmax=9)

    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    else:
        fig = axs[0].figure

    for ax in axs:
        ax.clear()
        ax.axis("off")

    # Plot the matrices (input, attempt, solution)
    for i, matrix in enumerate([input_matrix, solution_matrix, attempt_matrix]):
        axs[i].imshow(matrix, cmap=cmap, norm=norm)
        axs[i].grid(True, which="both", color="lightgrey", linewidth=0.5)
        axs[i].set_xticks([x - 0.5 for x in range(1 + len(matrix[0]))])
        axs[i].set_yticks([x - 0.5 for x in range(1 + len(matrix))])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

    if solution_matrix == attempt_matrix:
        attempt_color = "green"
    else:
        attempt_color = "red"

    axs[0].set_title("Input")
    axs[1].set_title("Solution")
    axs[2].set_title(title, color=attempt_color)

    if show:
        plt.show()

    return fig


def plot_results(attempts: list[Attempt]):
    num_plots = sum(1 + len(response.challenge.train) for response in attempts)
    total_axes = num_plots * 3  # Each plot_grid needs 3 axes
    ncols = 3  # Each plot_grid spans 3 columns
    nrows = math.ceil(total_axes / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axs = axs.flatten() if hasattr(axs, "flatten") else [axs]

    ind = 0
    for i, result in enumerate(attempts):
        fixing_str = ""
        if result.fixing:
            fixing_str = f"#{len(result.fixing_ids)}, {result.fixing.id}"
        llm_config = result.config.llm_config
        prompt_config = result.config.prompt_config
        title = f"Test Attempt #{i + 1}{fixing_str}\n{pformat(llm_config)}\n{pformat(prompt_config)}\ncost_cents={result.cost_cents:.2f}\nattempt_id={result.id}\nid={result.challenge.id}"
        plot_grid(
            input_matrix=result.challenge.test[0].input,
            attempt_matrix=result.test_attempt,
            solution_matrix=result.challenge.test[0].output,
            title=title,
            axs=axs[ind : ind + 3],
            show=False,
        )
        ind += 3

        # Plot the examples as well
        for j in range(len(result.challenge.train)):
            title = f"Train Attempt #{j + 1}"
            plot_grid(
                input_matrix=result.challenge.train[j].input,
                attempt_matrix=result.train_attempts[j],
                solution_matrix=result.challenge.train[j].output,
                title=title,
                axs=axs[ind : ind + 3],
                show=False,
            )
            ind += 3

    # Remove any unused subplots
    for i in range(ind, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()
