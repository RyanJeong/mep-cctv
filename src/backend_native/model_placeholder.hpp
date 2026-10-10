// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_BACKEND_NATIVE_MODEL_PLACEHOLDER_HPP_
#define SRC_BACKEND_NATIVE_MODEL_PLACEHOLDER_HPP_

/**
 * @file model_placeholder.hpp
 * @brief Stand-in for the TVM-compiled model; detects nothing.
 *
 * Exposes the same Forward interface as model_tvm.hpp without TVM dependencies.
 * It ignores input and returns zeros. Use CCTV_MODEL=real for inference;
 * see models/README.md.
 */

#include <cstdint>
#include <vector>

#include "cctv/cctv.h"

namespace mep_cctv {
namespace backend_native {
namespace generated {

/**
 * @brief Return an all-zero YOLOv8 output tensor.
 *
 * @param input_nchw  Non-NULL NCHW tensor; pixel values are ignored.
 * @param num_elems   Input element count; ignored by the placeholder.
 * @return            Zero vector of length core::kOutputStride *
 * core::kNumAnchors.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Allocates the output vector.
 * @side_effects_what         std::vector heap allocation.
 * @side_effects_impact       core::kOutputStride * core::kNumAnchors floats.
 * @side_effects_alternatives Caller-supplied buffer would work but couples
 *                            caller to the tensor layout.
 */
std::vector<float> Forward(const float* input_nchw, int num_elems) noexcept;

}  // namespace generated
}  // namespace backend_native
}  // namespace mep_cctv

#endif  // SRC_BACKEND_NATIVE_MODEL_PLACEHOLDER_HPP_
