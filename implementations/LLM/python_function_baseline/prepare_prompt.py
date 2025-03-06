from data_loading import training_challenges
from examples import example_1_same_grid_challenge_id, example_1_same_grid_reasoning

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