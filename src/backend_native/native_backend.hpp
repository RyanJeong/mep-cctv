// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_BACKEND_NATIVE_NATIVE_BACKEND_HPP_
#define SRC_BACKEND_NATIVE_NATIVE_BACKEND_HPP_

/**
 * @file native_backend.hpp
 * @brief Native TVM backend interface.
 *
 * Wraps the TVM-compiled (placeholder) model functions and exposes the
 * backend interface consumed by the C ABI layer.
 */

#include <vector>

#include "cctv/cctv.h"
#include "src/core/image_ops.hpp"

namespace mep_cctv {
namespace backend_native {

/**
 * @brief Native TVM backend: runs the compiled (placeholder) model.
 *
 * Immutable after construction: all inference state is read-only.
 */
class NativeBackend {
 public:
  /**
   * @brief Construct the backend with inference thresholds.
   *
   * @param conf_threshold  Minimum detection score (0..1).
   * @param iou_threshold   NMS IoU threshold (0..1).
   *
   * @side_effects None -- referentially transparent.
   */
  explicit NativeBackend(float conf_threshold, float iou_threshold) noexcept;

  /**
   * @brief Run object detection on a preprocessed float tensor.
   *
   * @param nchw_tensor  Float[1 * 3 * 640 * 640] NCHW tensor.
   * @param num_elems    Must equal 1*3*640*640.
   * @param params       Letterbox params used in preprocessing.
   * @param orig_w       Original image width.
   * @param orig_h       Original image height.
   * @return             Detected boxes in original image coordinates.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason       Allocates heap vectors for model output and
   *                            boxes.
   * @side_effects_what         std::vector heap allocations inside called
   *                            functions.
   * @side_effects_impact       O(kOutputElements + detections); ~2.8 MB
   *                            transient.
   * @side_effects_alternatives Pre-allocated buffers would reduce alloc
   *                            pressure but complicate the ownership model.
   */
  std::vector<cctv_box> Detect(const float* nchw_tensor, int num_elems,
                               const core::LetterboxParams& params, int orig_w,
                               int orig_h) const noexcept;

 private:
  const float conf_threshold_;
  const float iou_threshold_;
};

}  // namespace backend_native
}  // namespace mep_cctv

#endif  // SRC_BACKEND_NATIVE_NATIVE_BACKEND_HPP_
