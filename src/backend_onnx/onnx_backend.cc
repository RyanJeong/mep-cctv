// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file onnx_backend.cc
 * @brief ONNX Runtime backend implementation.
 *
 * YOLOv8 inference pipeline:
 *   letterbox(rgb) -> float NCHW tensor -> ORT session -> [1,84,kNumAnchors]
 *   -> DecodeYoloOutput -> NMS -> cctv_box list
 *
 * Detect catches exceptions and returns an empty result; it does not
 * distinguish inference failure from a frame with no detections.
 */

#include "src/backend_onnx/onnx_backend.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Keep the third-party C++ header after standard headers despite its .h suffix.
#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_subdir,build/include_order)

#include "src/core/image_ops.hpp"
#include "src/core/yolo_postprocess.hpp"

namespace mep_cctv {
namespace backend_onnx {

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

/**
 * @brief Build ORT session options with automatic intra-op threading.
 *
 * @return SessionOptions with graph optimizations enabled.
 *
 * @side_effects None -- referentially transparent.
 */
Ort::SessionOptions MakeSessionOptions() noexcept {
  Ort::SessionOptions opts;
  // Let ONNX Runtime choose intra-op thread count from available CPU cores.
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

    // Allocated-name accessors support the pinned ONNX Runtime 1.18.1.
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

  // The environment must be constructed before, and outlive, the session.
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
    // YOLO input: [1, 3, kInputSize, kInputSize] float32.
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

    // Expected output: [1, kOutputStride, kNumAnchors] float32.
    const float* raw = outputs[0].GetTensorData<float>();
    return core::DecodeYoloOutput(raw, params, orig_w, orig_h, conf_threshold_,
                                  iou_threshold_);
  } catch (...) {
    // Current failure contract returns no detections.
    return {};
  }
}

}  // namespace backend_onnx
}  // namespace mep_cctv
