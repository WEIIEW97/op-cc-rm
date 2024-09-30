import numpy as np
from scipy.ndimage import label
from numba import jit


def op_cc_4n_thr_removal_v2(m, radius, thr):
    padded_m = np.pad(m, pad_width=radius, mode="edge")
    h, w = m.shape

    for i in range(h):
        for j in range(w):
            patch = padded_m[i : i + 2 * radius + 1, j : j + 2 * radius + 1]
            # Create a mask for connected components

            mask = np.abs(patch - patch[radius, radius]) < thr

            # Label connected components

            labeled, n_labels = label(mask)

            for label_id in range(1, n_labels + 1):
                region = labeled == label_id
                if (
                    np.any(region[0, :])
                    or np.any(region[-1, :])
                    or np.any(region[:, 0])
                    or np.any(region[:, -1])
                ):
                    continue
                patch[region] = 0
            padded_m[i : i + 2 * radius + 1, j : j + 2 * radius + 1] = patch
    return padded_m[radius : h + radius, radius : w + radius]


def op_cc_4n_thr_removal_v3(m, radius, thr):
    padded_m = np.pad(m, pad_width=radius, mode='edge')
    h, w = m.shape

    # Define the DFS function to find and clear connected components
    def dfs(x, y, component, visited, patch, radius, thr):
        # Stack for the DFS
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if not visited[cx, cy]:
                visited[cx, cy] = True
                component.append((cx, cy))
                # Check the four neighboring cells
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < patch.shape[0] and 0 <= ny < patch.shape[1]:
                        if not visited[nx, ny] and abs(patch[nx, ny] - patch[radius, radius]) < thr:
                            stack.append((nx, ny))

    for i in range(h):
        for j in range(w):
            patch = padded_m[i:i + 2 * radius + 1, j:j + 2 * radius + 1]
            visited = np.zeros_like(patch, dtype=bool)
            mask = np.abs(patch - patch[radius, radius]) < thr
            
            for r in range(patch.shape[0]):
                for c in range(patch.shape[1]):
                    if mask[r, c] and not visited[r, c]:
                        component = []
                        dfs(r, c, component, visited, patch, radius, thr)
                        # Check if the component is closed
                        if all(not (x == 0 or x == patch.shape[0] - 1 or y == 0 or y == patch.shape[1] - 1) for x, y in component):
                            for x, y in component:
                                patch[x, y] = 0
            
            padded_m[i:i + 2 * radius + 1, j:j + 2 * radius + 1] = patch

    return padded_m[radius:h + radius, radius:w + radius]