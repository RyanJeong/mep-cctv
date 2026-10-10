// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file nms_scalar.cc
 * @brief Scalar (portable C++14) implementation of NMS.
 */

#include <algorithm>
#include <vector>

#include "src/core/nms.hpp"

namespace mep_cctv {
namespace core {

float ComputeIoU(const cctv_box& a, const cctv_box& b) noexcept {
  const float ax1 = a.x;
  const float ay1 = a.y;
  const float ax2 = a.x + a.w;
  const float ay2 = a.y + a.h;

  const float bx1 = b.x;
  const float by1 = b.y;
  const float bx2 = b.x + b.w;
  const float by2 = b.y + b.h;

  const float inter_x1 = std::max(ax1, bx1);
  const float inter_y1 = std::max(ay1, by1);
  const float inter_x2 = std::min(ax2, bx2);
  const float inter_y2 = std::min(ay2, by2);

  const float inter_w = inter_x2 - inter_x1;
  const float inter_h = inter_y2 - inter_y1;
  if (inter_w <= 0.0f || inter_h <= 0.0f) return 0.0f;

  const float inter_area = inter_w * inter_h;
  const float area_a = a.w * a.h;
  const float area_b = b.w * b.h;
  const float union_area = area_a + area_b - inter_area;
  if (union_area <= 0.0f) return 0.0f;

  return inter_area / union_area;
}

std::vector<cctv_box> ApplyNMS(const std::vector<cctv_box>& boxes,
                               float iou_threshold) noexcept {
  if (boxes.empty()) return {};

  // Sort by descending score.
  std::vector<cctv_box> sorted = boxes;
  std::sort(
      sorted.begin(), sorted.end(),
      [](const cctv_box& a, const cctv_box& b) { return a.score > b.score; });

  std::vector<bool> suppressed(sorted.size(), false);
  std::vector<cctv_box> result;

  for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
    if (suppressed[static_cast<unsigned>(i)]) continue;
    result.push_back(sorted[static_cast<unsigned>(i)]);
    for (int j = i + 1; j < static_cast<int>(sorted.size()); ++j) {
      if (suppressed[static_cast<unsigned>(j)]) continue;
      if (ComputeIoU(sorted[static_cast<unsigned>(i)],
                     sorted[static_cast<unsigned>(j)]) > iou_threshold) {
        suppressed[static_cast<unsigned>(j)] = true;
      }
    }
  }

  return result;
}

}  // namespace core
}  // namespace mep_cctv
