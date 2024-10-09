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

#include "src/cc.h"
#include <Eigen/Core>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <stack>
#include <string>

using std::string;

const vector<vector<int>> Direction::four = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

int main() {
  string data_path =
      "/home/william/Codes/op-kernel-cc-thr-rm/data/dispL_x64_U16.png";
  int radius = 5;
  float thr = 2.5;
  cv::Mat m_cv = cv::imread(data_path, cv::IMREAD_ANYDEPTH);
  m_cv.convertTo(m_cv, CV_32F);

  Eigen::MatrixXf m;
  cv::cv2eigen(m_cv, m);
  m /= 64;
  Eigen::MatrixXf mp;
  mp = op_cc_4n_thr_removal(m, radius, thr);

  cv::Mat m_cv_n, mp_cv_n;
  cv::normalize(m_cv, m_cv_n, 0, 1, cv::NORM_MINMAX);
  cv::Mat m_cv_8u;
  m_cv_n.convertTo(m_cv_8u, CV_8UC1, 255.0);
  cv::Mat m_cv_color;
  cv::applyColorMap(m_cv_8u, m_cv_color, cv::COLORMAP_MAGMA);

  cv::Mat mp_cv;
  cv::eigen2cv(mp, mp_cv);
  cv::normalize(mp_cv, mp_cv_n, 0, 1, cv::NORM_MINMAX);
  cv::Mat mp_cv_8u;
  mp_cv_n.convertTo(mp_cv_8u, CV_8UC1, 255.0);
  cv::Mat mp_cv_color;
  cv::applyColorMap(mp_cv_8u, mp_cv_color, cv::COLORMAP_MAGMA);

  cv::namedWindow("before Image", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("after Image", cv::WINDOW_AUTOSIZE);

  //// Show the images
  cv::imshow("before Image", m_cv_color);
  cv::imshow("after Image", mp_cv_color);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}