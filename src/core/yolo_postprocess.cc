// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file yolo_postprocess.cc
 * @brief Implementation of YOLOv8 output tensor decoding and NMS.
 */

#include "src/core/yolo_postprocess.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "src/core/nms.hpp"

namespace mep_cctv {
namespace core {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

/** @brief One raw YOLO box, still in letterbox pixel coordinates. */
struct LetterboxBox {
  float cx;
  float cy;
  float w;
  float h;
};

/** @brief Dimensions of the original, pre-letterbox image. */
struct ImageSize {
  int w;
  int h;
};

/** @brief Map center-based letterbox coordinates to original-image box extents.
 */
cctv_box DecodeBox(const LetterboxBox& raw, float score, int class_id,
                   const LetterboxParams& params, ImageSize orig) noexcept {
  // Half-padding offsets (same formula as ResizeLetterbox).
  const float off_x = params.pad_w * 0.5f;
  const float off_y = params.pad_h * 0.5f;

  // Remove padding, then undo scaling to get original pixel coords.
  const float cx_orig = (raw.cx - off_x) / params.scale;
  const float cy_orig = (raw.cy - off_y) / params.scale;
  const float w_orig = raw.w / params.scale;
  const float h_orig = raw.h / params.scale;

  // Clamp to image boundaries.
  const float x = std::max(0.0f, cx_orig - w_orig * 0.5f);
  const float y = std::max(0.0f, cy_orig - h_orig * 0.5f);
  const float x2 =
      std::min(static_cast<float>(orig.w), cx_orig + w_orig * 0.5f);
  const float y2 =
      std::min(static_cast<float>(orig.h), cy_orig + h_orig * 0.5f);

  cctv_box box;
  box.x = x;
  box.y = y;
  // Independently clipped edges can cross for boxes outside the image.
  box.w = std::max(0.0f, x2 - x);
  box.h = std::max(0.0f, y2 - y);
  box.score = score;
  box.class_id = class_id;
  return box;
}

}  // namespace

// ---------------------------------------------------------------------------
// DecodeYoloOutput
// ---------------------------------------------------------------------------

std::vector<cctv_box> DecodeYoloOutput(const float* raw_output,
                                       const LetterboxParams& params,
                                       int orig_w, int orig_h,
                                       float conf_threshold,
                                       float iou_threshold) noexcept {
  assert(raw_output != nullptr && orig_w > 0 && orig_h > 0);
  assert(params.scale > 0.0f);

  std::vector<cctv_box> candidates;

  // raw_output layout: [kOutputStride][kNumAnchors]
  // Row 0..3:  box coords (cx, cy, w, h) in letterbox pixels
  // Row 4..83: class probabilities, already sigmoid-normalized by the export.
  for (int a = 0; a < kNumAnchors; ++a) {
    // Find the maximum class score for this anchor.
    float max_score = 0.0f;
    int best_class = 0;
    for (int c = 0; c < kNumClasses; ++c) {
      const float s = raw_output[(4 + c) * kNumAnchors + a];
      if (s > max_score) {
        max_score = s;
        best_class = c;
      }
    }

    if (max_score < conf_threshold) continue;

    const float cx = raw_output[0 * kNumAnchors + a];
    const float cy = raw_output[1 * kNumAnchors + a];
    const float bw = raw_output[2 * kNumAnchors + a];
    const float bh = raw_output[3 * kNumAnchors + a];

    const LetterboxBox raw = {cx, cy, bw, bh};
    const ImageSize orig = {orig_w, orig_h};
    candidates.push_back(DecodeBox(raw, max_score, best_class, params, orig));
  }

  if (candidates.empty()) return {};

  return ApplyNMS(candidates, iou_threshold);
}

}  // namespace core
}  // namespace mep_cctv
