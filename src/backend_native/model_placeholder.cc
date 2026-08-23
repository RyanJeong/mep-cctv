// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file model_placeholder.cc
 * @brief Synthetic stand-in for the TVM-compiled model.
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
// Constants for synthetic output
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

  // All zeros, which decodes to no detections at any confidence threshold.
  //
  // This used to inject two convincing boxes so the annotated stream had
  // something to show. That made a build without the real model look like a
  // working detector: the page drew labelled rectangles around objects that
  // were never there, and nothing on screen said otherwise. Reporting
  // nothing is the honest failure mode -- the pipeline, the encoder and the
  // page are all still exercised, and an empty frame is a visible prompt to
  // run the conversion step rather than a convincing lie.
  return std::vector<float>(static_cast<unsigned>(kOutputElements), 0.0f);
}

}  // namespace generated
}  // namespace backend_native
}  // namespace mep_cctv
