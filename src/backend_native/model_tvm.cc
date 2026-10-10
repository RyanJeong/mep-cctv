// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file model_tvm.cc
 * @brief Runs the TVM-compiled YOLOv8n embedded in this library.
 *
 * See model_tvm.hpp for the contract and for how the generated translation
 * units get here.
 */

#include "src/backend_native/model_tvm.hpp"

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "src/core/image_ops.hpp"
#include "src/core/yolo_postprocess.hpp"

namespace mep_cctv {
namespace backend_native {
namespace generated {

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Keep native tensor geometry aligned with shared preprocessing and decoding.
constexpr int kInputElements = core::kInputElements;
constexpr int kOutputElements = core::kOutputStride * core::kNumAnchors;
constexpr int64_t kInputDim = core::kInputSize;

// The Relax VM allocator kind: 2 selects the pooled allocator, which reuses
// intermediate buffers across calls instead of malloc/free per layer.
constexpr int kPooledAllocator = 2;

/**
 * @brief Minimal host allocator for input tensors.
 *
 * malloc/free rather than new/delete, and deliberately so: DLTensor::data is
 * a void* in the DLPack C ABI, TVM decides when FreeData runs, and malloc
 * guarantees alignment suitable for any scalar type. Nothing here can own the
 * pointer in the RAII sense -- ownership lives in a C struct that TVM manages
 * -- so a smart pointer has nowhere to live.
 */
struct CpuAlloc {
  void AllocData(DLTensor* tensor) {
    // NOSONAR cpp:S1231 -- C allocation is correct for a DLPack C-ABI hook.
    tensor->data = /* NOSONAR */ std::malloc(tvm::ffi::GetDataSize(*tensor));
  }

  // NOSONAR cpp:S1231 -- must pair with the malloc above.
  void FreeData(DLTensor* tensor) { /* NOSONAR */ std::free(tensor->data); }
};

/**
 * @brief Build the Relax VM around the linked-in model.
 *
 * ffi.SystemLib resolves symbols registered by the embedded model at load time.
 */
tvm::ffi::Module CreateVirtualMachine() {
  const auto system_lib =
      tvm::ffi::Function::GetGlobalRequired("ffi.SystemLib");
  auto executable = system_lib().cast<tvm::ffi::Module>();
  auto vm = executable->GetFunction("vm_load_executable")
                .value()()
                .cast<tvm::ffi::Module>();
  vm->GetFunction("vm_initialization")
      .value()(static_cast<int>(kDLCPU), 0, kPooledAllocator,
               static_cast<int>(kDLCPU), 0, kPooledAllocator);
  return vm;
}

/**
 * @brief Return the process-wide VM, constructing it on first use.
 *
 * Function-local statics are initialised exactly once, and C++11 onwards
 * guarantees that initialisation is thread-safe.  Concurrent *calls* into the
 * returned VM are not guaranteed safe; callers must serialize inference.
 */
tvm::ffi::Module& CachedVirtualMachine() {
  static tvm::ffi::Module vm = CreateVirtualMachine();
  return vm;
}

}  // namespace

// ---------------------------------------------------------------------------
// Forward (visual)
// ---------------------------------------------------------------------------

std::vector<float> Forward(const float* input_nchw, int num_elems) noexcept {
  std::vector<float> output(static_cast<unsigned>(kOutputElements), 0.0f);
  if (input_nchw == nullptr || num_elems != kInputElements) return output;

  try {
    const DLDevice device{kDLCPU, 0};
    const std::vector<int64_t> shape{1, 3, kInputDim, kInputDim};
    auto input =
        tvm::ffi::Tensor::FromNDAlloc(CpuAlloc(), tvm::ffi::Shape(shape),
                                      DLDataType{kDLFloat, 32, 1}, device);

    auto* dst = static_cast<float*>(input.data_ptr());
    for (int i = 0; i < kInputElements; ++i) dst[i] = input_nchw[i];

    auto& vm = CachedVirtualMachine();
    const auto result =
        vm->GetFunction("main").value()(input).cast<tvm::ffi::Tensor>();

    const auto* src = static_cast<const float*>(result.data_ptr());
    for (int i = 0; i < kOutputElements; ++i)
      output[static_cast<unsigned>(i)] = src[i];
  } catch (...) {
    // Current failure contract returns zeros rather than propagating TVM
    // errors.
    output.assign(static_cast<unsigned>(kOutputElements), 0.0f);
  }
  return output;
}

}  // namespace generated
}  // namespace backend_native
}  // namespace mep_cctv
