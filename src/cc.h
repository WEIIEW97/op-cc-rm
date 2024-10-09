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

struct Direction {
  const std::vector<std::vector<int>> four = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
};

// padding a Eigen::Matrix in edge replicate mode
template <typename Derived>
Eigen::Matrix<Derived, -1, -1> pad(const Eigen::Matrix<Derived, -1, -1> &M,
                                   int pad_width) {
  Eigen::Matrix<Derived, -1, -1> MP(M.rows() + 2 * pad_width,
                                    M.cols() + 2 * pad_width);
  int H, W, HP, WP;
  H = M.rows(), W = M.cols(), HP = MP.rows(), WP = MP.cols();
  MP.block(pad_width, pad_width, M.rows(), M.cols()) = M;
  for (int i = 0; i < pad_width; ++i) {
    MP.row(i).head(WP) = MP.row(pad_width);
    MP.row(HP - 1 - i) = MP.row(HP - 1 - pad_width);
    MP.col(i) = MP.col(pad_width);
    MP.col(WP - 1 - i) = MP.col(WP - 1 - pad_width);
  }
  return MP;
}

template <typename Derived>
void dfs(int x, int y, std::vector<Eigen::Vector2i> &component,
         Eigen::Matrix<bool, -1, -1> &visited,
         const Eigen::Matrix<Derived, -1, -1> &patch, int radius, float thr) {
  std::stack<Eigen::Vector2i> stack;
  stack.push(Eigen::Vector2i(x, y));
  int ph = patch.rows();
  int pw = patch.cols();
  while (!stack.empty()) {
    auto _coord = stack.top();
    stack.pop();
    int cx = _coord(0), cy = _coord(1);
    if (!visited(cx, cy)) {
      visited(cx, cy) = true;
      component.push_back(_coord);
      for (auto &dir : Direction().four) {
        int dx = dir[0], dy = dir[1];
        int nx = cx + dx, ny = cy + dy;
        if (nx >= 0 && nx < ph && ny >= 0 && ny < pw) {
          if (!visited(nx, ny) &&
              std::abs(patch(nx, ny) - patch(radius, radius)) < thr)
            stack.push(Eigen::Vector2i(nx, ny));
        }
      }
    }
  }
}

bool check_closed(const std::vector<Eigen::Vector2i> &components, int x_bound,
                  int y_bound) {
  for (const auto &coord : components) {
    int x = coord(0), y = coord(1);
    if (x == 0 || x == x_bound - 1 || y == 0 || y == y_bound - 1)
      return false;
  }
  return true;
}

template <typename Derived>
Eigen::Matrix<Derived, -1, -1>
op_cc_4n_thr_removal(const Eigen::Matrix<Derived, -1, -1> &m, int radius,
                     float thr) {
  auto MP = pad(m, radius);
  int h = m.rows(), w = m.cols();

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      auto patch = MP.block(i, j, i + 2 * radius + 1, j + 2 * radius + 1);
      Eigen::Matrix<bool, -1, -1> visited(patch.rows(), patch.cols());
      visited.setZero();
      Eigen::Matrix<Derived, -1, -1> c_patch = Eigen::Matrix<Derived, -1, -1>::Constant(patch.rows(), patch.cols(), patch(radius, radius));
      auto mask = (Eigen::abs(patch - c_patch).array() < thr).matrix();
      for (int r = 0; r < patch.rows(); ++r) {
        for (int c = 0; c < patch.cols(); ++c) {
          if (mask(r, c) && !visited(r, c)) {
            std::vector<Eigen::Vector2i> component;
            dfs(r, c, component, visited, patch, radius, thr);
            // check if the component is closed
            if (check_closed(component, patch.rows(), patch.cols())) {
              for (const auto &coord : component) {
                patch(coord(0), coord(1)) = 0;
              }
            }
          }
        }
      }
      MP.block(i, j, i + 2 * radius + 1, j + 2 * radius + 1) = patch;
    }
  }

  return MP.block(radius, radius, h, w);
}