// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file native_backend.cc
 * @brief NativeBackend implementation.
 */

#include "src/backend_native/native_backend.hpp"

#include <vector>

// Both headers declare the same Forward() in the same
// namespace, so the call sites below are identical either way.  CMake defines
// CCTV_USE_GENERATED_MODEL=1 when the TVM-compiled model has been generated
// and linked in; otherwise the synthetic placeholder keeps the tree buildable.
#if defined(CCTV_USE_GENERATED_MODEL) && CCTV_USE_GENERATED_MODEL
#include "src/backend_native/model_tvm.hpp"
#else
#include "src/backend_native/model_placeholder.hpp"
#endif

#include "src/core/yolo_postprocess.hpp"

namespace mep_cctv {
namespace backend_native {

NativeBackend::NativeBackend(float conf_threshold, float iou_threshold) noexcept
    : conf_threshold_(conf_threshold), iou_threshold_(iou_threshold) {}

std::vector<cctv_box> NativeBackend::Detect(const float* nchw_tensor,
                                            int num_elems,
                                            const core::LetterboxParams& params,
                                            int orig_w,
                                            int orig_h) const noexcept {
  const std::vector<float> raw = generated::Forward(nchw_tensor, num_elems);

  return core::DecodeYoloOutput(raw.data(), params, orig_w, orig_h,
                                conf_threshold_, iou_threshold_);
}

}  // namespace backend_native
}  // namespace mep_cctv
