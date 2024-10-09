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

#include <Eigen/Dense>
#include <cmath>
#include <stack>
#include <vector>
#include <omp.h>

using std::vector, std::stack;
using Eigen::Matrix, Eigen::Vector2i, Eigen::Dynamic, Eigen::MatrixBase;

struct Direction {
  static const vector<vector<int>> four;
};

// Pad a matrix in edge replicate mode
template <typename Derived>
Matrix<Derived, Dynamic, Dynamic>
pad(const Matrix<Derived, Dynamic, Dynamic>& M, int pad_width) {
  Matrix<Derived, Dynamic, Dynamic> MP(M.rows() + 2 * pad_width,
                                       M.cols() + 2 * pad_width);
  int H = M.rows(), W = M.cols(), HP = MP.rows(), WP = MP.cols();
  MP.block(pad_width, pad_width, H, W) = M;
  for (int i = 0; i < pad_width; ++i) {
    MP.row(i).head(WP) = MP.row(pad_width);
    MP.row(HP - 1 - i) = MP.row(HP - pad_width - 1);
    MP.col(i) = MP.col(pad_width);
    MP.col(WP - 1 - i) = MP.col(WP - pad_width - 1);
  }
  return MP;
}

template <typename Derived>
void dfs(int x, int y, vector<Vector2i>& component,
         Matrix<bool, Dynamic, Dynamic>& visited,
         const MatrixBase<Derived>& patch, int radius, float thr) {
  stack<Vector2i> stack;
  stack.emplace(x, y);
  while (!stack.empty()) {
    auto coord = stack.top();
    stack.pop();
    int cx = coord[0], cy = coord[1];
    if (!visited(cx, cy)) {
      visited(cx, cy) = true;
      component.emplace_back(cx, cy);
      for (const auto& dir : Direction::four) {
        int nx = cx + dir[0], ny = cy + dir[1];
        if (nx >= 0 && nx < patch.rows() && ny >= 0 && ny < patch.cols()) {
          if (!visited(nx, ny) &&
              abs(patch(nx, ny) - patch(radius, radius)) < thr)
            stack.emplace(nx, ny);
        }
      }
    }
  }
}

bool check_closed(const vector<Vector2i>& components, int x_bound,
                  int y_bound) {
  for (const auto& coord : components) {
    if (coord[0] == 0 || coord[0] == x_bound - 1 || coord[1] == 0 ||
        coord[1] == y_bound - 1)
      return false;
  }
  return true;
}

template <typename Derived>
Matrix<Derived, Dynamic, Dynamic>
op_cc_4n_thr_removal(const Matrix<Derived, Dynamic, Dynamic>& m, int radius,
                     float thr) {
  auto MP = pad(m, radius);
  int h = m.rows(), w = m.cols();
  
#pragma omp parallel for collapse(2)
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto patch = MP.block(i, j, 2 * radius + 1, 2 * radius + 1);
      Matrix<bool, Dynamic, Dynamic> visited(patch.rows(), patch.cols());
      visited.setZero();
      auto c_patch = Matrix<Derived, Dynamic, Dynamic>::Constant(
          patch.rows(), patch.cols(), patch(radius, radius));
      auto mask = (Eigen::abs(patch.array() - c_patch.array()) < thr).matrix();
      for (int r = 0; r < patch.rows(); ++r) {
        for (int c = 0; c < patch.cols(); ++c) {
          if (mask(r, c) && !visited(r, c)) {
            vector<Vector2i> component;
            dfs(r, c, component, visited, patch, radius, thr);
            if (check_closed(component, patch.rows(), patch.cols())) {
              for (const auto& coord : component) {
                patch(coord[0], coord[1]) = 0;
              }
            }
          }
        }
      }
      MP.block(i, j, 2 * radius + 1, 2 * radius + 1) = patch;
    }
  }
  return MP.block(radius, radius, h + radius, w + radius);
}
