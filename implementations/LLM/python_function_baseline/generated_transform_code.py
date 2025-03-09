def transform(grid):
    # work on a copy so as not to change the original
    H = len(grid)
    W = len(grid[0])
    new_grid = [row[:] for row in grid]
    
    # only work on the lower half: rows starting at floor(H/2)
    lower_start = H // 2
    
    # helper: get 4-connected component starting from (r,c)
    def get_component(r, c, visited):
        comp = []
        stack = [(r, c)]
        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in visited:
                continue
            visited.add((rr, cc))
            comp.append((rr, cc))
            for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr, nc = rr+dr, cc+dc
                if lower_start <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == 2 and (nr, nc) not in visited:
                        stack.append((nr, nc))
        return comp

    # get all positions in lower half with value 2
    positions = [(r, c) for r in range(lower_start, H) for c in range(W) if grid[r][c] == 2]
    if not positions:
        return new_grid

    # identify connected components among the 2's
    visited = set()
    comps = []
    for r, c in positions:
        if (r, c) not in visited:
            comp = get_component(r, c, visited)
            comps.append(comp)
    
    # compute for each component its bounding box and center
    comp_infos = []
    for comp in comps:
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        info = {
            'min_r': min(rows),
            'max_r': max(rows),
            'min_c': min(cols),
            'max_c': max(cols),
            'center_r': sum(rows)/len(rows),
            'center_c': sum(cols)/len(cols),
            'cells': comp
        }
        comp_infos.append(info)
    # sort by horizontal position (min_c)
    comp_infos.sort(key=lambda info: info['min_c'])
    
    # If there is only one component, check if it is "wide"
    if len(comp_infos) == 1:
        info = comp_infos[0]
        width = info['max_c'] - info['min_c']
        # For a wide component we “cast” a shadow on its left.
        if width >= 1:
            # For every row in the vertical span of the component,
            # change all 0’s to 1’s that lie to the left of the component.
            for r in range(info['min_r'], info['max_r']+1):
                for c in range(0, info['min_c']):
                    if new_grid[r][c] == 0:
                        new_grid[r][c] = 1
        else:
            # For a narrow component (essentially a single column of 2’s), draw a narrow horizontal band.
            band_row = int(round(info['center_r']))
            for c in range(max(0, int(info['center_c'])-1), min(W, int(info['center_c'])+2)):
                if new_grid[band_row][c] == 0:
                    new_grid[band_row][c] = 1
            connector_row = band_row + 1
            if connector_row < H and new_grid[connector_row][int(info['center_c'])] == 0:
                new_grid[connector_row][int(info['center_c'])] = 1
    else:
        # If there are two (or more) components in the lower half, process each separately.
        # In the examples the behavior is:
        #  – In any row where a component’s 2’s appear as two separate segments,
        #    fill the gap between them with 1’s.
        #  – Then for each component, draw a horizontal “band” (of width 3, centered on the component)
        #    in one row (the band row is chosen as one row below the top of the component) and a “connector”
        #    (a single 1 at the center) in the following row.
        #  – Finally, below the groups, a “base‐line” is drawn as a horizontal band spanning
        #    from a couple cells to the right of the leftmost 2 to a couple cells to the left of the rightmost 2.
        
        # First, for any row in the lower half that contains two separated segments of 2’s,
        # fill the gap between them with 1’s.
        for r in range(lower_start, H):
            # find contiguous runs (segments) of 2 in row r
            segments = []
            c = 0
            while c < W:
                if grid[r][c] == 2:
                    start = c
                    while c < W and grid[r][c] == 2:
                        c += 1
                    segments.append((start, c-1))
                else:
                    c += 1
            if len(segments) >= 2:
                # fill the gap between the right‐end of the first and the left–start of the second
                # (assume two groups per row as in our examples)
                seg1, seg2 = segments[0], segments[1]
                for cc in range(seg1[1]+1, seg2[0]):
                    if new_grid[r][cc] == 0:
                        new_grid[r][cc] = 1
        
        # now process each component individually (assuming two components in examples)
        for info in comp_infos:
            # choose a “band” row roughly one row below the component’s top
            band_row = info['min_r'] + 1
            # for a narrow component (a single column of 2’s), fill a band of width 3
            if info['max_c'] - info['min_c'] < 1:
                c_center = int(round(info['center_c']))
                for c in range(max(0, c_center-1), min(W, c_center+2)):
                    if new_grid[band_row][c] == 0:
                        new_grid[band_row][c] = 1
                connector_row = band_row + 1
                if connector_row < H and new_grid[connector_row][c_center] == 0:
                    new_grid[connector_row][c_center] = 1
            else:
                # for a component that is wider, we “bridge” or “connect” its parts.
                # In the vertical span of the component, for any row that has two 2‐segments,
                # the gap has already been bridged above.
                # Also, below the component we add a narrow connector.
                c_center = int(round(info['center_c']))
                for c in range(max(0, c_center-1), min(W, c_center+2)):
                    if new_grid[band_row][c] == 0:
                        new_grid[band_row][c] = 1
                connector_row = band_row + 1
                if connector_row < H and new_grid[connector_row][c_center] == 0:
                    new_grid[connector_row][c_center] = 1
        # Finally, draw a base‐line a couple rows below the lower–most row among the components.
        base_row = max(info['max_r'] for info in comp_infos) + 2
        if base_row < H:
            left_bound = min(info['min_c'] for info in comp_infos) + 2
            right_bound = max(info['max_c'] for info in comp_infos) + 1
            for c in range(left_bound, min(right_bound+1, W)):
                if new_grid[base_row][c] == 0:
                    new_grid[base_row][c] = 1

    return new_grid