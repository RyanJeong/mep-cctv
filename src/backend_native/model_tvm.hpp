// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_BACKEND_NATIVE_MODEL_TVM_HPP_
#define SRC_BACKEND_NATIVE_MODEL_TVM_HPP_

/**
 * @file model_tvm.hpp
 * @brief Real TVM-compiled YOLOv8n inference for the native backend.
 *
 * Drop-in replacement for model_placeholder.hpp, selected by the CMake option
 * CCTV_MODEL=real.  Unlike the placeholder, Forward() runs the
 * actual network.
 *
 * Nothing is read from disk at run time.  The fused kernels and the weights
 * are linked into libcctv from two generated files
 * (produced by models/convert/tvm_compile.py --emit-embed):
 *
 *   yolov8n_tuned_kernels.o  target-specific LLVM-generated kernels
 *   yolov8n_tuned_blob.cc    weights and self-registering Relax VM bytecode
 *
 * At first use the module is fetched from TVM's system-library registry and a
 * Relax VM is created around it.  This costs one lazy initialisation; every
 * later call reuses it.
 *
 * This file is hand-written and committed. The generated files above
 * are gitignored -- regenerate them before building with
 * CCTV_MODEL=real.
 */

#include <cstdint>
#include <vector>

#include "cctv/cctv.h"

namespace mep_cctv {
namespace backend_native {
namespace generated {

/**
 * @brief Run the TVM-compiled YOLOv8n forward pass.
 *
 * @param input_nchw  Float[core::kInputElements] NCHW tensor, letterboxed and
 *                    normalised to [0, 1].  Must not be NULL.
 * @param num_elems   Must equal core::kInputElements.
 * @return            Float vector of length core::kOutputStride *
 * core::kNumAnchors, the raw pre-NMS YOLOv8 head output.  Returns an all-zero
 * vector of the same size if inference fails, so callers always receive a
 *                    well-formed tensor.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Lazily constructs a process-wide Relax VM on first
 *                            call and allocates the output vector.
 * @side_effects_what         One-time VM construction; std::vector heap
 *                            allocation.
 * @side_effects_impact       First call pays VM setup; later calls are
 *                            steady-state. The cached VM is not thread-safe for
 *                            concurrent calls.
 * @side_effects_alternatives Constructing the VM per call would repeat the
 *                            setup cost on every frame, which dominates
 * runtime.
 */
std::vector<float> Forward(const float* input_nchw, int num_elems) noexcept;

}  // namespace generated
}  // namespace backend_native
}  // namespace mep_cctv

#endif  // SRC_BACKEND_NATIVE_MODEL_TVM_HPP_
