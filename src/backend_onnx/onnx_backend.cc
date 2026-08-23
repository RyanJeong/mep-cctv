// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file onnx_backend.cc
 * @brief ONNX Runtime dl backend implementation.
 *
 * YOLOv8 inference pipeline:
 *   letterbox(rgb) -> float NCHW tensor -> ORT session -> [1,84,8400]
 *   -> DecodeYoloOutput -> NMS -> cctv_box list
 *
 *
 * All ORT calls may throw Ort::Exception.  Detect catches all
 * exceptions internally so that callers receive a safe empty result rather
 * than an unhandled exception propagating through the C ABI.
 */

#include "src/backend_onnx/onnx_backend.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// The ONLY translation unit that sees ONNX Runtime. Its C API puts exception
// specifications in typedefs, which C++14 rejects outright, so this file
// alone is compiled as C++17 (see CMakeLists.txt).
//
// build/include_order is suppressed because cpplint reads a .h in angle
// brackets as a C system header and wants it above the C++ ones. This is a
// third-party C++ header, so its place here -- after the standard library and
// before the project headers -- is what the project's own include order
#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_subdir,build/include_order)

#include "src/core/image_ops.hpp"
#include "src/core/yolo_postprocess.hpp"

namespace mep_cctv {
namespace backend_onnx {

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

/**
 * @brief Build a default single-threaded ORT session options struct.
 *
 * @return SessionOptions using every core, with WARN log level.
 *
 * @side_effects None -- referentially transparent.
 */
Ort::SessionOptions MakeSessionOptions() noexcept {
  Ort::SessionOptions opts;
  // 0 means "use all available cores". This backend was previously pinned to
  // a single thread, which made the onnx-versus-native comparison unfair:
  // the native model runs its autotuned schedules across every core, so
  // handicapping ONNX Runtime would have flattered native for no good reason.
  // Both modes now get the whole CPU.
  opts.SetIntraOpNumThreads(0);

  // ORT_ENABLE_ALL turns on the layout and fusion passes that make ORT fast
  // on ARM.
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return opts;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl -- everything that touches ONNX Runtime types
// ---------------------------------------------------------------------------

class OnnxBackend::Impl {
 public:
  explicit Impl(const char* yolo_path) {
    Ort::SessionOptions opts = MakeSessionOptions();
    session = Ort::Session(env, yolo_path, opts);

    // Use the ...NameAllocated() accessors rather than GetInputNames(): the
    // vector-returning form only exists in newer ONNX Runtime (1.27 on a
    // Homebrew Mac), while the Raspberry Pi build pins 1.18.1, where it does
    // not. The allocator form is present in every version we support.
    Ort::AllocatorWithDefaultOptions allocator;
    if (session.GetInputCount() == 0 || session.GetOutputCount() == 0)
      throw OnnxBackendError("OnnxBackend: YOLO model has no I/O nodes");

    const Ort::AllocatedStringPtr in =
        session.GetInputNameAllocated(0, allocator);
    const Ort::AllocatedStringPtr out =
        session.GetOutputNameAllocated(0, allocator);
    input_name = std::string(in.get());
    output_name = std::string(out.get());
  }

  // Both take only constants, so they are initialised in-class: the values
  // belong to the member, not to one particular constructor. Declaration
  // order still matters -- env must outlive and precede session, which is
  // assigned from it in the body.
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mep_cctv"};
  Ort::Session session{nullptr};
  std::string input_name;
  std::string output_name;
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

OnnxBackend::OnnxBackend(const char* yolo_path, float conf_threshold,
                         float iou_threshold)
    : conf_threshold_(conf_threshold), iou_threshold_(iou_threshold) {
  if (yolo_path == nullptr)
    throw OnnxBackendError("OnnxBackend: yolo_model_path must not be NULL");

  impl_.reset(new Impl(yolo_path));
}

OnnxBackend::~OnnxBackend() = default;

// ---------------------------------------------------------------------------
// Detect (YOLO)
// ---------------------------------------------------------------------------

std::vector<cctv_box> OnnxBackend::Detect(const float* tensor,
                                          const core::LetterboxParams& params,
                                          int orig_w, int orig_h) noexcept {
  try {
    // YOLO input: [1, 3, 640, 640] float32.
    const std::array<int64_t, 4> input_shape{1, 3, core::kInputSize,
                                             core::kInputSize};

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(tensor),  // NOSONAR - ORT requires non-const ptr
        static_cast<size_t>(core::kInputElements), input_shape.data(),
        input_shape.size());

    const char* input_name = impl_->input_name.c_str();
    const char* output_name = impl_->output_name.c_str();

    Ort::RunOptions run_opts{nullptr};
    std::vector<Ort::Value> outputs = impl_->session.Run(
        run_opts, &input_name, &input_tensor, 1, &output_name, 1);

    if (outputs.empty()) return {};

    // Output tensor shape: [1, 84, 8400] float32.
    const float* raw = outputs[0].GetTensorData<float>();
    return core::DecodeYoloOutput(raw, params, orig_w, orig_h, conf_threshold_,
                                  iou_threshold_);
  } catch (...) {
    // Swallow ORT exceptions; return empty detections rather than crashing.
    return {};
  }
}

}  // namespace backend_onnx
}  // namespace mep_cctv
