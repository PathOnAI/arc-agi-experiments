from copy import deepcopy


def transform_gpt_4o(grid: list[list[int]]) -> list[list[int]]:
    def expand_3x3_to_9x9(subgrid):
        expanded = [[0] * 9 for _ in range(9)]

        # Set values according to observed patterns
        for i in range(3):
            for j in range(3):
                value = subgrid[i][j]
                if value != 0:
                    # Expand this value into 5 specific cells in the 9x9 grid
                    positions = [
                        (i * 3, j * 3),
                        (i * 3 + 1, j * 3 + 1),
                        (i * 3 + 2, j * 3 + 2),
                        (i * 3 + 2, j * 3),
                        (i * 3, j * 3 + 2),
                    ]
                    for x, y in positions:
                        expanded[x][y] = value
        return expanded

    # Expand input grid
    output_grid = expand_3x3_to_9x9(grid)
    return output_grid


import numpy as np


def transform_claude(grid: list[list[int]]) -> list[list[int]]:
    input_grid = np.array(grid)
    output_grid = np.zeros((9, 9), dtype=int)

    # Copy input to corners
    output_grid[:3, :3] = input_grid
    output_grid[:3, 6:] = input_grid
    output_grid[6:, :3] = input_grid
    output_grid[6:, 6:] = input_grid

    # Vertical flips for top and bottom middle
    output_grid[:3, 3:6] = np.flip(input_grid, axis=0)
    output_grid[6:, 3:6] = np.flip(input_grid, axis=0)

    # Horizontal flips for left and right middle
    output_grid[3:6, :3] = np.flip(input_grid, axis=1)
    output_grid[3:6, 6:] = np.flip(input_grid, axis=1)

    return output_grid.tolist()


def transform_leg(grid_lst: list[list[int]]) -> list[list[int]]:
    # Convert input to numpy array for easier manipulation
    input_grid = np.array(grid_lst)

    # Create 9x9 output grid
    output_grid = np.zeros((9, 9), dtype=int)

    # Helper function to place 3x3 grid into 9x9 grid
    def place_subgrid(row, col, subgrid):
        output_grid[row * 3 : (row + 1) * 3, col * 3 : (col + 1) * 3] = subgrid

    # Create transformed grids
    rotated = np.rot90(input_grid, 2)
    mirrored_vertical = np.flipud(input_grid)
    mirrored_horizontal = np.fliplr(input_grid)

    # Place subgrids
    # Corners (rotated)
    place_subgrid(0, 0, rotated)
    place_subgrid(0, 2, rotated)
    place_subgrid(2, 0, rotated)
    place_subgrid(2, 2, rotated)

    # Edges (mirrored)
    place_subgrid(0, 1, mirrored_vertical)
    place_subgrid(1, 0, mirrored_horizontal)
    place_subgrid(1, 2, mirrored_horizontal)
    place_subgrid(2, 1, mirrored_vertical)

    # Center (original)
    place_subgrid(1, 1, input_grid)

    return output_grid.tolist()


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    # Initialize a 9x9 output grid with zeros
    output_grid = [[0 for _ in range(9)] for _ in range(9)]

    for i in range(3):
        for j in range(3):
            color = input_grid[i][j]
            if color != 0:
                # Fill the corresponding 3x3 subgrid in the output
                for x in range(3):
                    for y in range(3):
                        if input_grid[x][y] != 0:
                            output_grid[i * 3 + x][j * 3 + y] = color

    return output_grid


def transform_old(grid: list[list[int]]) -> list[list[int]]:
    input_grid = np.array(grid)
    output_grid = np.zeros((9, 9), dtype=int)

    # Define the rotations for each corner
    rotations = {
        (0, 0): lambda x: x,  # Top-left: No rotation
        (0, 2): lambda x: np.rot90(x, k=-1),  # Top-right: 90 degrees clockwise
        (2, 0): lambda x: np.rot90(x, k=1),  # Bottom-left: 90 degrees counterclockwise
        (2, 2): lambda x: np.rot90(x, k=2),  # Bottom-right: 180 degrees
    }

    for i in range(3):
        for j in range(3):
            if input_grid[i, j] != 0:
                # Determine the rotation based on the position
                if (i, j) in rotations:
                    rotated_input = rotations[(i, j)](input_grid)
                else:
                    # For the center cell, we'll just use an empty 3x3 grid
                    rotated_input = np.zeros((3, 3), dtype=int)

                # Fill the corresponding 3x3 block in the output grid
                output_grid[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] = (
                    rotated_input != 0
                ) * input_grid[i, j]

    return output_grid.tolist()


def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    # Define the replication pattern for each input cell
    replication_pattern = [
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # Top-left
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # Top-center
        [[0, 0, 1], [0, 0, 0], [0, 0, 0]],  # Top-right
        [[0, 1, 0], [0, 1, 0], [0, 0, 0]],  # Middle-left
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Middle-center
        [[0, 1, 0], [0, 1, 0], [0, 0, 0]],  # Middle-right
        [[0, 0, 0], [0, 0, 0], [1, 0, 0]],  # Bottom-left
        [[0, 0, 0], [0, 1, 0], [0, 1, 0]],  # Bottom-center
        [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # Bottom-right
    ]

    output_grid = [[0 for _ in range(9)] for _ in range(9)]

    for i in range(3):
        for j in range(3):
            cell_value = grid_lst[i][j]
            pattern = replication_pattern[i * 3 + j]
            for x in range(3):
                for y in range(3):
                    if pattern[x][y] == 1:
                        output_grid[i * 3 + x][j * 3 + y] = cell_value

    return output_grid


def rotate(grid, times):
    return np.rot90(grid, times)


def transform(input_grid: list[list[int]]) -> list[list[int]]:
    input_array = np.array(input_grid)
    output = np.zeros((9, 9), dtype=int)

    rotations = [[0, 1, 2], [3, 0, 1], [2, 3, 0]]

    for i in range(3):
        for j in range(3):
            if input_array[i, j] != 0:
                rotated = rotate(input_array, rotations[i][j])
                output[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = rotated * (
                    input_array[i, j] != 0
                )

    return output.tolist()


from scipy.ndimage import find_objects, label


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert the input grid to a numpy array
    grid = np.array(grid)

    # Find connected components of green cells
    green_cells = grid == 3
    labeled_array, num_features = label(green_cells)

    # Process each connected component
    for i in range(1, num_features + 1):
        component = labeled_array == i
        slice_y, slice_x = find_objects(component)[0]

        height = slice_y.stop - slice_y.start
        width = slice_x.stop - slice_x.start

        if height < 3 or width < 3:
            continue

        # Calculate the size of the yellow square
        if height <= 4 and width <= 4:
            yellow_size = 2
        else:
            yellow_size = min(height, width) - 2

        # Calculate the position to place the yellow square
        start_y = slice_y.start + (height - yellow_size) // 2
        start_x = slice_x.start + (width - yellow_size) // 2

        # Place the yellow square
        grid[start_y : start_y + yellow_size, start_x : start_x + yellow_size] = 4

    return grid.tolist()


from scipy.ndimage import generate_binary_structure


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert input to numpy array
    grid = np.array(grid)

    # Create a mask of green cells (value 3)
    green_mask = grid == 3

    # Create a structure that includes diagonal connections
    s = generate_binary_structure(2, 2)

    # Invert the green mask to get potential fill regions
    fill_regions = ~green_mask

    # Label the fill regions
    labeled_regions, num_regions = label(fill_regions, structure=s)

    # Create output grid, starting with the input grid
    output = grid.copy()

    # Iterate through each labeled region
    for region in range(1, num_regions + 1):
        region_mask = labeled_regions == region

        # Check if the region is completely enclosed by green cells
        if not np.any(
            region_mask & ~np.pad(green_mask, 1, mode="constant", constant_values=True)
        ):
            # Check if the region is at least 2x2 in size
            if np.sum(region_mask) >= 4:
                # Fill the region with yellow (4)
                output[region_mask] = 4

    return output.tolist()


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert the input grid to a numpy array
    arr = np.array(grid)

    # Find all green regions
    green_mask = arr == 3
    labeled, num_features = label(green_mask)

    # Process each region
    for i in range(1, num_features + 1):
        region = labeled == i
        slice_x, slice_y = find_objects(region)[0]

        # Check if the region is 2x2 or larger
        if slice_x.stop - slice_x.start >= 2 and slice_y.stop - slice_y.start >= 2:
            # Create a mask for the interior of the region
            interior = np.zeros_like(region)
            interior[slice_x, slice_y] = 1
            interior[slice_x.start, slice_y] = 0
            interior[slice_x.stop - 1, slice_y] = 0
            interior[slice_x, slice_y.start] = 0
            interior[slice_x, slice_y.stop - 1] = 0

            # Fill the interior with yellow (4)
            arr[interior == 1] = 4

    return arr.tolist()


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert the input list to a numpy array for easier manipulation
    arr = np.array(grid)
    rows, cols = arr.shape

    # Helper function to check if a cell is part of a green rectangle
    def is_green_rectangle(r, c):
        if arr[r, c] != 3:
            return False
        width = height = 1
        while c + width < cols and arr[r, c + width] == 3:
            width += 1
        while r + height < rows and all(arr[r + height, c : c + width] == 3):
            height += 1
        return (width > 1 or height > 1), height, width

    # Iterate through the grid
    for r in range(rows):
        for c in range(cols):
            is_rect, height, width = is_green_rectangle(r, c)
            if is_rect:
                # For 2x2, change all to yellow
                if height == 2 and width == 2:
                    arr[r : r + 2, c : c + 2] = 4
                # For larger rectangles, change inner cells to yellow
                elif height > 1 and width > 1:
                    arr[r + 1 : r + height - 1, c + 1 : c + width - 1] = 4

    return arr.tolist()


def transform(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    def is_square(r, c, size):
        if r + size > rows or c + size > cols:
            return False
        return all(
            grid[i][j] == 3 for i in range(r, r + size) for j in range(c, c + size)
        )

    def find_largest_square(r, c):
        size = 1
        while is_square(r, c, size + 1):
            size += 1
        return size

    def fill_square(r, c, size):
        for i in range(r + 1, r + size - 1):
            for j in range(c + 1, c + size - 1):
                grid[i][j] = 4

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                size = find_largest_square(r, c)
                if size >= 2:
                    fill_square(r, c, size)

    return grid


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert the input grid to a numpy array
    array = np.array(grid)

    # Find all green shapes
    green_shapes = array == 3
    labeled_array, num_features = label(green_shapes)

    # Process each green shape
    for i in range(1, num_features + 1):
        shape = labeled_array == i
        slice_y, slice_x = find_objects(shape)[0]

        height = slice_y.stop - slice_y.start
        width = slice_x.stop - slice_x.start

        # Calculate yellow area dimensions
        yellow_height = max(0, height - 2)
        yellow_width = max(0, width - 2)

        # Fill the center with yellow if the shape is 2x2 or larger
        if yellow_height > 0 and yellow_width > 0:
            y_start = slice_y.start + (height - yellow_height) // 2
            x_start = slice_x.start + (width - yellow_width) // 2
            array[
                y_start : y_start + yellow_height, x_start : x_start + yellow_width
            ] = 4

    return array.tolist()


def transform(grid: list[list[int]]) -> list[list[int]]:
    rows, cols = len(grid), len(grid[0])

    def find_rectangle(r, c):
        max_r, max_c = r, c
        while max_r + 1 < rows and grid[max_r + 1][c] == 3:
            max_r += 1
        while max_c + 1 < cols and grid[r][max_c + 1] == 3:
            max_c += 1
        return max_r, max_c

    def fill_yellow(r1, c1, r2, c2):
        height, width = r2 - r1 + 1, c2 - c1 + 1
        if height == 2 and width == 2:
            grid[r1][c1] = 4
        elif height > 2 and width > 2:
            for i in range(r1 + 1, r2):
                for j in range(c1 + 1, c2):
                    grid[i][j] = 4

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                max_r, max_c = find_rectangle(r, c)
                if max_r > r and max_c > c:
                    fill_yellow(r, c, max_r, max_c)

    return grid


from scipy import ndimage


def transform(grid):
    grid = np.array(grid)
    green_mask = grid == 3
    labeled_array, num_features = ndimage.label(green_mask)

    for label in range(1, num_features + 1):
        shape = labeled_array == label
        shape_size = np.sum(shape)

        if shape_size <= 9:  # 3x3 or smaller
            yellow_size = 1
        elif shape_size <= 25:  # 5x5 or smaller
            yellow_size = 4  # 2x2 square
        else:
            yellow_size = shape_size // 2

        distance_transform = ndimage.distance_transform_edt(shape)
        max_dist = np.max(distance_transform)

        yellow_mask = (distance_transform >= max_dist - 0.5) & (
            distance_transform <= max_dist
        )
        yellow_cells = np.argwhere(yellow_mask)

        sorted_indices = np.argsort(
            [-distance_transform[tuple(cell)] for cell in yellow_cells]
        )
        yellow_cells = yellow_cells[sorted_indices]

        for i in range(min(yellow_size, len(yellow_cells))):
            y, x = yellow_cells[i]
            if np.all(
                grid[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] != 0
            ):  # Ensure surrounded by non-black cells
                grid[y, x] = 4  # Change to yellow

    return grid.tolist()


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert the input list to a numpy array for easier processing
    arr = np.array(grid)

    # Find all contiguous areas of green (3) cells
    green_mask = arr == 3
    labeled, num_features = label(green_mask)

    # For each contiguous area
    for i in range(1, num_features + 1):
        area = labeled == i
        coords = np.argwhere(area)

        # Check if it's a perfect square
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)
        height = max_row - min_row + 1
        width = max_col - min_col + 1

        if height == width and np.all(
            area[min_row : max_row + 1, min_col : max_col + 1]
        ):
            # If it's a perfect square, change to yellow (4)
            arr[area] = 4

    # Convert back to list and return
    return arr.tolist()


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert the input grid to a numpy array
    arr = np.array(grid)

    # Create a binary mask of green cells
    green_mask = arr == 3

    # Label connected components in the green mask
    labeled, num_features = label(green_mask)

    # Create a copy of the input array for the output
    output = arr.copy()

    # Iterate through all labeled components
    for i in range(1, num_features + 1):
        # Get the bounding box of the current component
        bbox = find_objects(labeled == i)[0]

        # Check if the component is at least 2x2 in size
        if (bbox[0].stop - bbox[0].start >= 2) and (bbox[1].stop - bbox[1].start >= 2):
            # Check if the component forms a rectangle
            component = green_mask[bbox]
            if np.all(component):
                # Fill the rectangle with yellow (4) in the output
                output[bbox][component] = 4

    # Convert the output array back to a list of lists
    return output.tolist()


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert the input grid to a numpy array
    arr = np.array(grid)

    # Find all connected components of green cells
    green_mask = arr == 3
    labeled_array, num_features = label(green_mask)

    # Process each connected component
    for i in range(1, num_features + 1):
        component = labeled_array == i
        slice_obj = find_objects(component)[0]

        # Extract the component
        sub_arr = arr[slice_obj]

        # Find the largest rectangle in this component
        height, width = sub_arr.shape
        max_area = 0
        max_rect = None

        for y in range(height):
            for x in range(width):
                for h in range(1, height - y + 1):
                    for w in range(1, width - x + 1):
                        rect = sub_arr[y : y + h, x : x + w]
                        if np.all(rect == 3) and h * w > max_area:
                            # Check if the rectangle is surrounded by green
                            if y > 0 and np.any(sub_arr[y - 1, x : x + w] != 3):
                                continue
                            if y + h < height and np.any(
                                sub_arr[y + h, x : x + w] != 3
                            ):
                                continue
                            if x > 0 and np.any(sub_arr[y : y + h, x - 1] != 3):
                                continue
                            if x + w < width and np.any(sub_arr[y : y + h, x + w] != 3):
                                continue
                            max_area = h * w
                            max_rect = (y, x, h, w)

        # Fill the largest rectangle with yellow
        if max_rect:
            y, x, h, w = max_rect
            sub_arr[y : y + h, x : x + w] = 4

        # Update the original array
        arr[slice_obj] = sub_arr

    return arr.tolist()


def transform(grid: list[list[int]]) -> list[list[int]]:
    # Convert the input grid to a numpy array
    arr = np.array(grid)

    # Create a mask of green cells
    green_mask = arr == 3

    # Label connected components of green cells
    labeled, num_features = label(green_mask)

    # For each connected component
    for i in range(1, num_features + 1):
        component = labeled == i
        slice_x, slice_y = find_objects(component)[0]

        # Check if the component is at least 2x2
        if slice_x.stop - slice_x.start >= 2 and slice_y.stop - slice_y.start >= 2:
            # Create a mask for the inner cells
            inner = component.copy()
            inner[slice_x.start, :] = False
            inner[slice_x.stop - 1, :] = False
            inner[:, slice_y.start] = False
            inner[:, slice_y.stop - 1] = False

            # Change inner cells to yellow
            arr[inner] = 4

    return arr.tolist()


from typing import List


def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid)
    height, width = grid.shape

    def is_valid(x, y):
        return 0 <= x < height and 0 <= y < width

    def has_2x2_green(x, y):
        if not is_valid(x + 1, y + 1):
            return False
        return np.all(grid[x : x + 2, y : y + 2] == 3)

    def flood_fill(x, y):
        if not is_valid(x, y) or grid[x, y] != 3:
            return
        grid[x, y] = 4
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            flood_fill(x + dx, y + dy)

    # Find 2x2 green squares and flood fill
    for x in range(height):
        for y in range(width):
            if has_2x2_green(x, y):
                flood_fill(x, y)

    # Restore the green outline
    for x in range(height):
        for y in range(width):
            if grid[x, y] == 4:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny) and grid[nx, ny] == 0:
                        grid[x, y] = 3
                        break

    return grid.tolist()


from collections import deque
from typing import Tuple


def transform(grid: list[list[int]]) -> list[list[int]]:
    def find_connected_shape(start_r: int, start_c: int) -> list[tuple[int, int]]:
        shape = []
        queue = deque([(start_r, start_c)])
        visited = set([(start_r, start_c)])

        while queue:
            r, c = queue.popleft()
            shape.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < len(grid)
                    and 0 <= nc < len(grid[0])
                    and grid[nr][nc] == 3
                    and (nr, nc) not in visited
                ):
                    queue.append((nr, nc))
                    visited.add((nr, nc))

        return shape

    def find_center_and_max_square(
        shape: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], int]:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)

        center_r = (min_r + max_r) // 2
        center_c = (min_c + max_c) // 2

        max_size = min(
            center_r - min_r + 1,
            max_r - center_r + 1,
            center_c - min_c + 1,
            max_c - center_c + 1,
        )

        while max_size > 1:
            all_green = all(
                grid[r][c] == 3
                for r in range(center_r - max_size + 1, center_r + max_size)
                for c in range(center_c - max_size + 1, center_c + max_size)
            )
            if all_green:
                break
            max_size -= 1

        return (center_r, center_c), max_size

    # Create a copy of the input grid
    new_grid = [row[:] for row in grid]

    # Find and transform all connected green shapes
    visited = set()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 3 and (r, c) not in visited:
                shape = find_connected_shape(r, c)
                visited.update(shape)

                center, size = find_center_and_max_square(shape)

                if size >= 2:
                    for dr in range(-size + 1, size):
                        for dc in range(-size + 1, size):
                            new_grid[center[0] + dr][center[1] + dc] = 4

    return new_grid


def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    green_color = 3
    yellow_color = 4

    # Identify green cells
    green_mask = grid == green_color

    # Invert and label the grid to find enclosed zero components
    inverted_grid = ~green_mask
    labeled_grid, num_features = label(inverted_grid)

    rows, cols = grid.shape

    for component in range(1, num_features + 1):
        component_mask = labeled_grid == component

        if is_fully_enclosed(component_mask, green_mask, rows, cols):
            grid[component_mask] = yellow_color

    return grid.tolist()


def is_fully_enclosed(component_mask, green_mask, rows, cols):
    # Check if all border cells in the component are surrounded by green
    for r in range(rows):
        for c in range(cols):
            if component_mask[r, c]:
                # Scan neighbors of the border cells
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not green_mask[nr, nc]:
                            return False
    return True


from scipy.ndimage import center_of_mass


def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    green_color = 3
    yellow_color = 4

    # Identify regions of green (color 3)
    labeled, num_features = label(grid == green_color)

    # For each region, calculate its center of mass and change it to yellow (color 4)
    for i in range(1, num_features + 1):
        com = center_of_mass(grid, labeled, i)
        center_row, center_col = int(round(com[0])), int(round(com[1]))
        if labeled[center_row, center_col] == i:
            grid[center_row, center_col] = yellow_color

    return grid.tolist()


def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    import numpy as np
    from scipy.ndimage import label

    # Convert the grid to a numpy array for convenience
    grid = np.array(grid_lst)
    fill_value = 4  # Yellow

    # Find areas that are considered 'holes' internally using a complementary approach
    # Mark green areas (3) as background and check connected components of non-green areas
    not_green = (grid != 3).astype(int)

    # Find connected components in the non-green area
    labeled_array, num_features = label(not_green)

    # For each connected component, if it is totally enclosed by green, fill it
    for feature in range(1, num_features + 1):
        feature_mask = labeled_array == feature

        # Check if the component is fully enclosed
        border_touching = (
            np.any(feature_mask[0, :])
            or np.any(feature_mask[-1, :])
            or np.any(feature_mask[:, 0])
            or np.any(feature_mask[:, -1])
        )

        if not border_touching:
            # Fill the component with yellow
            grid[feature_mask] = fill_value

    return grid.tolist()


import copy


def transform(grid_lst):
    grid = np.array(grid_lst)
    rows, cols = grid.shape
    new_grid = copy.deepcopy(grid)

    def is_green(cell):
        return cell == 3

    def get_neighbors(row, col):
        neighbors = [
            (row - 1, col - 1),  # top left
            (row - 1, col),  # top
            (row - 1, col + 1),  # top right
            (row, col - 1),  # left
            (row, col + 1),  # right
            (row + 1, col - 1),  # bottom left
            (row + 1, col),  # bottom
            (row + 1, col + 1),  # bottom right
        ]
        valid_neighbors = []
        for r, c in neighbors:
            if 0 <= r < rows and 0 <= c < cols:
                valid_neighbors.append((r, c))
        return valid_neighbors

    def dfs(row, col):
        visited = {(row, col)}
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            for nr, nc in get_neighbors(r, c):
                if is_green(grid[nr, nc]) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

        return visited

    def has_bridge(visited):
        min_row, min_col = min(visited)
        max_row, max_col = max(visited)

        min_row_set = {c for r, c in visited if r == min_row}
        max_row_set = {c for r, c in visited if r == max_row}
        min_col_set = {r for r, c in visited if c == min_col}
        max_col_set = {r for r, c in visited if c == max_col}

        if len(min_row_set) > 1 or len(max_row_set) > 1:
            return True
        if len(min_col_set) > 1 or len(max_col_set) > 1:
            return True
        return False

    green_shapes = []
    for row in range(rows):
        for col in range(cols):
            if is_green(grid[row, col]):
                green_shapes.append(dfs(row, col))

    for shape in green_shapes:
        if has_bridge(shape):
            for row, col in shape:
                new_grid[row, col] = 4

    for row in range(rows):
        for col in range(cols):
            if is_green(new_grid[row, col]):
                for r, c in get_neighbors(row, col):
                    if (
                        new_grid[r, c] == 0
                        and len(
                            [
                                (x, y)
                                for x, y in get_neighbors(r, c)
                                if is_green(new_grid[x, y])
                            ]
                        )
                        > 1
                    ):
                        new_grid[r, c] = 3

    return new_grid.tolist()


def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid_lst)
    output = np.zeros_like(grid)
    num_rows, num_cols = grid.shape

    # Identify unique colors/shapes
    unique_colors = np.unique(grid)
    if 0 in unique_colors:
        unique_colors = unique_colors[
            unique_colors != 0
        ]  # Exclude the background color

    for color in unique_colors:
        # Create a binary array for the current color
        binary_shape = (grid == color).astype(int)
        # Label connected components
        labeled_array, num_features = label(binary_shape)

        for i in range(1, num_features + 1):
            # Extract each shape
            shape = (labeled_array == i).astype(int)
            # Find the bounds of the shape
            cols_with_shape = np.any(shape, axis=0)
            if cols_with_shape.any():
                start_col = np.argmax(cols_with_shape)
                end_col = len(cols_with_shape) - np.argmax(cols_with_shape[::-1]) - 1

                # Shift the shape by one column to the right if possible
                if end_col + 1 < num_cols:
                    shifted_shape = np.roll(shape, shift=1, axis=1)
                    shifted_shape[:, start_col] = (
                        0  # Clear the old position to avoid overlaps
                    )
                    output += shifted_shape * color
                else:
                    output += shape * color

    return output.tolist()


def transform(grid):
    grid = np.array(grid)
    # Step 1: Create a binary image where 1 represents the shape (value 5)
    binary_image = (grid == 5).astype(np.uint8)
    # Step 2: Initialize the skeleton and edge images
    skeleton = np.zeros_like(binary_image)
    temp_image = binary_image.copy()
    # Define structuring elements for neighbor checking
    struct_elem = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    # Step 3: Iteratively remove edge pixels
    changed = True
    while changed:
        changed = False
        # Count neighbors
        neighbor_count = convolve2d(
            temp_image, struct_elem, mode="same", boundary="fill"
        )
        # Identify edge pixels (pixels with 1 or 0 neighbors)
        edge_pixels = ((neighbor_count >= 1) & (neighbor_count <= 7)) & (
            temp_image == 1
        )
        # Identify pixels whose removal would not disconnect the image
        removable = edge_pixels & not_end_point(temp_image)
        # Remove pixels
        if np.any(removable):
            temp_image[removable] = 0
            changed = True
    # The remaining pixels are the skeleton
    skeleton = temp_image
    # Assign output values
    output = np.zeros_like(grid)
    output[grid == 0] = 0
    output[(grid == 5) & (skeleton == 1)] = 2
    output[(grid == 5) & (skeleton == 0)] = 8
    return output.tolist()


# Helper function to compute convolution without scipy
def convolve2d(image, kernel, mode="same", boundary="fill"):
    m, n = kernel.shape
    if (m == n) and (m % 2 == 1):
        pad_size = m // 2
        if boundary == "fill":
            padded_image = np.pad(image, pad_size, mode="constant", constant_values=0)
        elif boundary == "wrap":
            padded_image = np.pad(image, pad_size, mode="wrap")
        else:
            padded_image = np.pad(image, pad_size, mode="constant", constant_values=0)
        result = np.zeros_like(image)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                region = padded_image[i : i + m, j : j + n]
                result[i, j] = np.sum(region * kernel)
        return result
    else:
        raise ValueError("Kernel must be square and have odd dimensions.")


def not_end_point(image):
    """
    Identify pixels that are not end points.
    End points have only one neighbor in the structure.
    """
    struct_elem = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = convolve2d(image, struct_elem, mode="same", boundary="fill")
    end_points = (neighbor_count == 1) & (image == 1)
    return ~end_points


def solve_first() -> None:
    grid_lst = [[7, 0, 7], [7, 0, 7], [7, 7, 0]]
    result = transform(grid_lst)
    correct = result == [
        [7, 0, 7, 0, 0, 0, 7, 0, 7],
        [7, 0, 7, 0, 0, 0, 7, 0, 7],
        [7, 7, 0, 0, 0, 0, 7, 7, 0],
        [7, 0, 7, 0, 0, 0, 7, 0, 7],
        [7, 0, 7, 0, 0, 0, 7, 0, 7],
        [7, 7, 0, 0, 0, 0, 7, 7, 0],
        [7, 0, 7, 7, 0, 7, 0, 0, 0],
        [7, 0, 7, 7, 0, 7, 0, 0, 0],
        [7, 7, 0, 7, 7, 0, 0, 0, 0],
    ]
    print("CORRECT:", correct)


def solve_sec() -> None:
    from src.main import plot_grid

    grid_lst = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    solution = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 3, 3, 3, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 4, 4, 4, 4, 3, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4, 4, 4, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 3, 4, 4, 4, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 4, 4, 4, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 3, 4, 3, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    result = transform(deepcopy(grid_lst))
    plot_grid(
        input_matrix=grid_lst,
        attempt_matrix=result,
        solution_matrix=solution,
        title="sec",
    )
    correct = result == solution
    print("CORRECT:", correct)


def solve_late() -> None:
    from src.main import plot_grid

    grid_lst = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    grid_lst = [
        [0, 0, 5, 5, 0, 5, 5, 5, 0, 0, 0],
        [0, 0, 5, 5, 0, 0, 5, 0, 0, 0, 0],
        [0, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0],
        [0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    solution = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 4, 4, 4, 4, 4, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 4, 4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    solution = [
        [0, 0, 8, 8, 0, 2, 2, 2, 0, 0, 0],
        [0, 0, 8, 8, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 2, 2, 8, 8, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 8, 8, 2, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    result = transform(deepcopy(grid_lst))
    plot_grid(
        input_matrix=grid_lst,
        attempt_matrix=result,
        solution_matrix=solution,
        title="late",
    )
    correct = result == solution
    print("CORRECT:", correct)


if __name__ == "__main__":
    solve_late()
