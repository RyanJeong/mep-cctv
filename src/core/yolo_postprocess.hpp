// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_CORE_YOLO_POSTPROCESS_HPP_
#define SRC_CORE_YOLO_POSTPROCESS_HPP_

/**
 * @file yolo_postprocess.hpp
 * @brief Decode raw YOLOv8 output tensor into cctv_box list.
 *
 * YOLOv8 outputs a [1, 84, kNumAnchors] tensor (no objectness score; 80 class
 * probabilities + 4 box coordinates per anchor).
 *
 * Processing steps:
 *   1. Iterate over the anchors.
 *   2. Compute max class score across 80 classes.
 *   3. Apply confidence threshold.
 *   4. Decode (cx,cy,w,h) -> (x,y,w,h) in letterbox space.
 *   5. Map coordinates back to original image space.
 *   6. Apply class-agnostic greedy NMS.
 */

#include <vector>

#include "cctv/cctv.h"
#include "src/core/image_ops.hpp"

namespace mep_cctv {
namespace core {

/**
 * @brief Number of anchor predictions output by the YOLOv8 head.
 *
 * The head predicts on three feature maps, at strides 8, 16 and 32, so the
 * count is (S/8)^2 + (S/16)^2 + (S/32)^2, which simplifies to 21*(S/32)^2.
 * Deriving it from kInputSize means the two can never disagree: at 640 this
 * is 8400, at 320 it is 2100.
 */
constexpr int kNumAnchors = 21 * (kInputSize / 32) * (kInputSize / 32);

/**
 * @brief Stride of the raw output tensor along the second axis (84 rows).
 */
constexpr int kOutputStride = kNumClasses + 4;  // 84

/**
 * @brief Decode a raw YOLOv8 output tensor into cctv_box list.
 *
 * @param raw_output    Pointer to float[1 * kOutputStride * kNumAnchors]
 *                      in row-major (C-contiguous) layout: index as
 *                      raw_output[row * kNumAnchors + anchor].
 * @param params        Letterbox parameters used during preprocessing
 *                      (needed to unmap coordinates to original space).
 * @param orig_w        Original image width  (pixels, > 0).
 * @param orig_h        Original image height (pixels, > 0).
 * @param conf_threshold Minimum score to retain a candidate.
 * @param iou_threshold  IoU threshold for NMS.
 * @return              Filtered, NMS-suppressed list of detections in
 *                      original image coordinate space.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Returns a heap-allocated vector.
 * @side_effects_what         std::vector heap allocation.
 * @side_effects_impact       Proportional to
 *                            (candidate count * sizeof(cctv_box)).
 * @side_effects_alternatives A fixed-size output array would cap detections
 *                            and lose precision under high-density scenes.
 */
std::vector<cctv_box> DecodeYoloOutput(const float* raw_output,
                                       const LetterboxParams& params,
                                       int orig_w, int orig_h,
                                       float conf_threshold,
                                       float iou_threshold) noexcept;

}  // namespace core
}  // namespace mep_cctv

#endif  // SRC_CORE_YOLO_POSTPROCESS_HPP_
