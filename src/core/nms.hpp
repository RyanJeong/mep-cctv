// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_CORE_NMS_HPP_
#define SRC_CORE_NMS_HPP_

/**
 * @file nms.hpp
 * @brief Non-Maximum Suppression over bounding boxes.
 *
 * Two compiled copies:
 *   nms_scalar.cc  -- portable C++14 reference
 *   nms_simd.cc    -- NEON / AVX2 vectorised IoU inner loop
 *
 * Both copies return identical results for the same input.
 */

#include <vector>

#include "cctv/cctv.h"

namespace mep_cctv {
namespace core {

/**
 * @brief Compute the Intersection-over-Union between two boxes.
 *
 * @param a  First box (x, y, w, h in pixel space).
 * @param b  Second box.
 * @return IoU value in [0, 1].
 *
 * @side_effects None -- referentially transparent.
 */
float ComputeIoU(const cctv_box& a, const cctv_box& b) noexcept;

/**
 * @brief Apply class-agnostic greedy NMS to score-filtered boxes.
 *
 * Boxes are processed in descending score order.  A box is suppressed if
 * its IoU with any already-kept box exceeds @p iou_threshold.
 *
 * @param boxes          Candidate boxes (any order; sorted internally).
 * @param iou_threshold  Suppression threshold in (0, 1].
 * @return Subset of @p boxes that survive NMS, in descending score order.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Returns a heap-allocated vector.
 * @side_effects_what         Heap allocation via std::vector constructor.
 * @side_effects_impact       Proportional to number of input boxes.
 * @side_effects_alternatives A caller-supplied output buffer would avoid
 *                            the allocation but forces callers to bound count.
 */
std::vector<cctv_box> ApplyNMS(const std::vector<cctv_box>& boxes,
                               float iou_threshold) noexcept;

}  // namespace core
}  // namespace mep_cctv

#endif  // SRC_CORE_NMS_HPP_
