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

#include "src/cc.cuh"

int main() {
    const int width = 1024;
    const int height = 1024;
    const int radius = 1;
    const float threshold = 0.5;

    float *h_data = new float[width * height];
    bool *h_visited = new bool[width * height]();

    // Initialize matrix data (example initialization)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_data[i * width + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *d_data;
    bool *d_visited;
    cudaMalloc(&d_data, width * height * sizeof(float));
    cudaMalloc(&d_visited, width * height * sizeof(bool));
    cudaMemcpy(d_data, h_data, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, width * height * sizeof(bool), cudaMemcpyHostToDevice);

    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    dim3 threads(16, 16);
    int sharedMemorySize = ((2 * radius + 1) * (2 * radius + 1) * sizeof(bool)) + (4 * (2 * radius + 1) * (2 * radius + 1) * sizeof(int));
    dfs_kernel<<<blocks, threads, sharedMemorySize>>>(d_data, d_visited, width, height, radius, threshold);

    cudaMemcpy(h_data, d_data, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_visited, d_visited, width * height * sizeof(bool), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_visited);
    delete[] h_data;
    delete[] h_visited;

    return 0;
}