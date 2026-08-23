// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_BACKEND_NATIVE_MODEL_PLACEHOLDER_HPP_
#define SRC_BACKEND_NATIVE_MODEL_PLACEHOLDER_HPP_

/**
 * @file model_placeholder.hpp
 * @brief Stand-in for the TVM-compiled model; detects nothing.
 *
 * Selected when no TVM-compiled model is linked in. It
 * exposes the same interface as model_tvm.hpp, so the project builds and its
 * tests run on a machine that has neither the TVM toolchain nor the
 * generated sources (those are ~68 MB and gitignored).
 *
 * Forward() returns a deterministic tensor containing two fixed detections.
 * That exercises the whole post-processing path -- decode, NMS, coordinate
 * mapping -- without any model file.
 *
 * WARNING: the output is SYNTHETIC. It ignores the input entirely, so the
 * boxes never move. Build with -DCCTV_MODEL=real for real
 * inference; see models/README.md.
 */

#include <cstdint>
#include <vector>

#include "cctv/cctv.h"

namespace mep_cctv {
namespace backend_native {
namespace generated {

/**
 * @brief Return a synthetic YOLOv8 output tensor.
 *
 * @param input_nchw  Float[1 * 3 * 640 * 640] NCHW tensor. Ignored: the
 *                    result does not depend on it.
 * @param num_elems   Must equal 1 * 3 * 640 * 640 = 1228800.
 * @return            Float vector of shape [1 * 84 * 8400].
 *                    Contains mostly zeros with two synthetic high-confidence
 *                    detections at fixed anchor positions (class 0, score 0.9).
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Allocates a heap vector sized ~2.8 MB.
 * @side_effects_what         std::vector heap allocation.
 * @side_effects_impact       Constant size (84 * 8400 * sizeof(float)).
 * @side_effects_alternatives Caller-supplied buffer would work but couples
 *                            caller to the tensor layout.
 */
std::vector<float> Forward(const float* input_nchw, int num_elems) noexcept;

}  // namespace generated
}  // namespace backend_native
}  // namespace mep_cctv

#endif  // SRC_BACKEND_NATIVE_MODEL_PLACEHOLDER_HPP_
