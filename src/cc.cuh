/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>

// Directions for the 4-connected neighbors
__constant__ int directions[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

__device__ bool check_closed(int* components, int num_components, int x_bound,
                             int y_bound) {
  for (int i = 0; i < num_components; i += 2) {
    int x = components[i];
    int y = components[i + 1];
    if (x == 0 || x == x_bound - 1 || y == 0 || y == y_bound - 1) {
      return false;
    }
  }
  return true;
}

// DFS kernel to identify and zero out closed components
__global__ void dfs_kernel(float* data, bool* visited_global, int width,
                              int height, int radius, float thr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
    extern __shared__ int shared[];
    bool* visited =
        (bool*)&shared[0]; // Shared memory for visited flags within patch
    int* stack =
        (int*)&visited[(2 * radius + 1) * (2 * radius + 1)]; // Stack for DFS

    // Initialize visited matrix for each patch
    for (int i = 0; i < (2 * radius + 1) * (2 * radius + 1); i++) {
      visited[i] = false;
    }

    int stackSize = 0;
    int num_components = 0;

    // Start DFS from the center of the patch
    stack[stackSize++] = radius; // x coordinate in patch
    stack[stackSize++] = radius; // y coordinate in patch

    while (stackSize > 0) {
      int cy = stack[--stackSize];
      int cx = stack[--stackSize];

      if (!visited[cy * (2 * radius + 1) + cx]) {
        visited[cy * (2 * radius + 1) + cx] = true; // Mark as visited

        // Store component for checking closure
        stack[num_components++] = cx;
        stack[num_components++] = cy;

        // Explore neighbors
        for (int dir = 0; dir < 4; ++dir) {
          int nx = cx + directions[dir][0];
          int ny = cy + directions[dir][1];
          if (nx >= 0 && nx < 2 * radius + 1 && ny >= 0 &&
              ny < 2 * radius + 1) {
            if (!visited[ny * (2 * radius + 1) + nx] &&
                std::abs(data[(y + ny - radius) * width + (x + nx - radius)] -
                         data[y * width + x]) < thr) {
              stack[stackSize++] = nx;
              stack[stackSize++] = ny;
            }
          }
        }
      }
    }

    if (check_closed(stack, num_components, 2 * radius + 1, 2 * radius + 1)) {
      for (int i = 0; i < num_components; i += 2) {
        int cx = stack[i];
        int cy = stack[i + 1];
        data[(y + cy - radius) * width + (x + cx - radius)] =
            0; // Zero out the component
      }
    }
  }
}
