// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file model_placeholder.cc
 * @brief Zero-output stand-in for the TVM-compiled model.
 *
 * See model_placeholder.hpp. Produces deterministic output so the full
 * pipeline can be exercised without a model file.
 */

#include "src/backend_native/model_placeholder.hpp"

#include <cassert>
#include <cmath>
#include <vector>

#include "src/core/image_ops.hpp"
#include "src/core/yolo_postprocess.hpp"

namespace mep_cctv {
namespace backend_native {
namespace generated {

// ---------------------------------------------------------------------------
// Output geometry
// ---------------------------------------------------------------------------

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Number of output elements: 1 * kOutputStride * kNumAnchors
constexpr int kOutputElements = (core::kNumClasses + 4) * core::kNumAnchors;

}  // namespace

// ---------------------------------------------------------------------------
// Forward (visual)
// ---------------------------------------------------------------------------

std::vector<float> Forward(const float* input_nchw, int num_elems) noexcept {
  assert(input_nchw != nullptr);
  (void) input_nchw;  // Placeholder: input is not used.
  (void) num_elems;

  // Zero probabilities produce no detections at valid positive thresholds.
  return std::vector<float>(static_cast<unsigned>(kOutputElements), 0.0f);
}

}  // namespace generated
}  // namespace backend_native
}  // namespace mep_cctv
