// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file cctv_c_api.cc
 * @brief Implementation of the public C ABI declared in cctv.h.
 *
 * Owns the cctv_engine struct (config + backend pointer).
 * All cctv_* functions validate arguments and delegate to the selected
 * backend.
 *
 * Detection arrays are allocated and released inside this library.
 * Callers release them with cctv_free_detections.
 */

#include <cassert>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "cctv/cctv.h"
#include "src/core/audio_detect.hpp"
#include "src/core/image_ops.hpp"
#include "src/core/yolo_postprocess.hpp"

#ifndef CCTV_BACKEND_NAME
#define CCTV_BACKEND_NAME "native"
#endif

#ifndef CCTV_VERSION
#define CCTV_VERSION "0.1.0"
#endif

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

#if defined(CCTV_USE_NATIVE_BACKEND)
#include "src/backend_native/native_backend.hpp"
#endif

#if defined(CCTV_USE_ONNX_BACKEND)
#include "src/backend_onnx/onnx_backend.hpp"
#endif

// ---------------------------------------------------------------------------
// Engine struct
// ---------------------------------------------------------------------------

struct cctv_engine {
  cctv_config config;

#if defined(CCTV_USE_NATIVE_BACKEND)
  mep_cctv::backend_native::NativeBackend backend;

  explicit cctv_engine(const cctv_config& cfg)
      : config(cfg), backend(cfg.conf_threshold, cfg.iou_threshold) {}

#elif defined(CCTV_USE_ONNX_BACKEND)
  mep_cctv::backend_onnx::OnnxBackend backend;

  explicit cctv_engine(const cctv_config& cfg)
      : config(cfg),
        backend(cfg.yolo_model_path, cfg.conf_threshold, cfg.iou_threshold) {}

#else
  explicit cctv_engine(const cctv_config& cfg) : config(cfg) {}
#endif
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

/**
 * @brief Maximum accepted image dimension (width or height) in pixels.
 *
 * Bounds accepted geometry; callers still supply a complete width*height*3
 * buffer.
 */
constexpr int kMaxImageDim = 65535;

/**
 * @brief Maximum accepted number of audio samples per call.
 *
 * Prevents analogous overflow in audio buffer index arithmetic.
 * 10 minutes at 192 kHz = 115 200 000 samples; 120 000 000 gives headroom.
 */
constexpr int kMaxAudioSamples = 120000000;

/**
 * Shallow-copy configuration. The model path pointer is not duplicated.
 */
cctv_config CopyConfig(const cctv_config* src) noexcept {
  cctv_config dst;
  dst.yolo_model_path = src->yolo_model_path;
  dst.conf_threshold = src->conf_threshold;
  dst.iou_threshold = src->iou_threshold;
  return dst;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" {

cctv_engine* cctv_engine_create(const cctv_config* cfg) {
  if (cfg == nullptr) return nullptr;

  // Replace out-of-range thresholds with defaults.
  cctv_config validated = CopyConfig(cfg);
  if (validated.conf_threshold <= 0.0f || validated.conf_threshold > 1.0f)
    validated.conf_threshold = 0.25f;
  if (validated.iou_threshold <= 0.0f || validated.iou_threshold > 1.0f)
    validated.iou_threshold = 0.45f;

  // OnnxBackend constructor throws std::runtime_error if model files are
  // missing or unreadable.  Catch here so the C ABI returns NULL on failure.
  try {
    auto engine = std::make_unique<cctv_engine>(validated);
    return engine.release();
  } catch (...) {
    // Construction failures are represented by a null engine handle.
    return nullptr;
  }
}

cctv_status cctv_engine_destroy(cctv_engine* engine) {
  std::unique_ptr<cctv_engine> owned(engine);
  return CCTV_OK;
}

// NOSONAR cpp:S995 -- engine cannot be pointer-to-const. NativeBackend::Detect
// is const, but OnnxBackend::Detect is not: Ort::Session::Run mutates the
// session's internal arena. One public signature has to serve both modes, so
// it takes the weaker guarantee. cctv_detect_audio, which does not mutate the
// engine, is declared const.
cctv_status cctv_detect_objects(/* NOSONAR */ cctv_engine* engine,
                                const uint8_t* rgb, int width, int height,
                                cctv_detections* out) {
  if (engine == nullptr || rgb == nullptr || out == nullptr)
    return CCTV_ERR_INVALID_ARG;

  if (width <= 0 || height <= 0) return CCTV_ERR_INVALID_ARG;

  if (width > kMaxImageDim || height > kMaxImageDim)
    return CCTV_ERR_INVALID_ARG;

  // 1. Preprocess: letterbox resize to float NCHW.
  const mep_cctv::core::LetterboxParams params =
      mep_cctv::core::ComputeLetterboxParams(width, height);
  const std::vector<float> tensor =
      mep_cctv::core::ResizeLetterbox(rgb, width, height, params);

  // 2. Backend forward + postprocess.
#if defined(CCTV_USE_NATIVE_BACKEND)
  const std::vector<cctv_box> boxes = engine->backend.Detect(
      tensor.data(), static_cast<int>(tensor.size()), params, width, height);
#elif defined(CCTV_USE_ONNX_BACKEND)
  const std::vector<cctv_box> boxes =
      engine->backend.Detect(tensor.data(), params, width, height);
#else
  const std::vector<cctv_box> boxes;
#endif

  // 3. Allocate the output array (caller frees via cctv_free_detections).
  out->count = static_cast<int>(boxes.size());
  if (out->count == 0) {
    out->boxes = nullptr;
    return CCTV_OK;
  }

  // Retain ownership until transfer; nothrow preserves CCTV_ERR_ALLOC on
  // failure.
  std::unique_ptr<cctv_box[]> owned(
      new (std::nothrow) cctv_box[static_cast<unsigned>(out->count)]);
  if (owned == nullptr) {
    out->count = 0;
    return CCTV_ERR_ALLOC;
  }

  std::memcpy(owned.get(), boxes.data(),
              static_cast<unsigned>(out->count) * sizeof(cctv_box));
  out->boxes = owned.release();
  return CCTV_OK;
}

cctv_status cctv_free_detections(cctv_detections* dets) {
  if (dets == nullptr) return CCTV_OK;
  // Reacquire ownership before clearing the caller's result.
  const std::unique_ptr<cctv_box[]> owned(dets->boxes);
  dets->boxes = nullptr;
  dets->count = 0;
  return CCTV_OK;
}

cctv_status cctv_detect_audio(const cctv_engine* engine, const float* samples,
                              int num_samples, int sample_rate,
                              cctv_audio_result* out) {
  if (engine == nullptr || samples == nullptr || out == nullptr)
    return CCTV_ERR_INVALID_ARG;

  if (num_samples <= 0 || sample_rate <= 0) return CCTV_ERR_INVALID_ARG;
  if (num_samples > kMaxAudioSamples) return CCTV_ERR_INVALID_ARG;

  // Audio uses the same DC-corrected RMS detector in both modes.
  (void) engine;
  *out = mep_cctv::core::DetectAcousticActivity(samples, num_samples);

  return CCTV_OK;
}

const char* cctv_version(void) { return CCTV_VERSION; }

const char* cctv_backend(void) { return CCTV_BACKEND_NAME; }

int cctv_simd_enabled(void) {
#if defined(CCTV_SIMD_ENABLED) && CCTV_SIMD_ENABLED
  return 1;
#else
  return 0;
#endif
}

}  // extern "C"
