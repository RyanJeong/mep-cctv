// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file cctv_c_api.cc
 * @brief Implementation of the public C ABI declared in cctv.h.
 *
 * Owns the cctv_engine struct (config + backend pointer).
 * All cctv_* functions validate arguments and delegate to the selected
 * backend.
 *
 * The boxes array is allocated with new[] and released by
 * cctv_free_detections. Both ends live in this library, so a C caller never
 * touches the allocator -- it only has to call the documented free function,
 * which is the contract cctv.h states. (An earlier comment here claimed
 * malloc was needed "so the caller does not need to link the C++ runtime";
 * that was never true, because the caller does not free the array itself.)
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

#if defined(CCTV_USE_DL_BACKEND)
#include "src/backend_dl/onnx_backend.hpp"
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

#elif defined(CCTV_USE_DL_BACKEND)
  mep_cctv::backend_dl::OnnxBackend backend;

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
 * Rejects attacker-supplied dimensions that would cause the index arithmetic
 * `sy * src_w * 3` (before the int64_t promotion in image_ops) to overflow
 * even in the C ABI layer.  65535 is well above 4K (3840 x 2160) so all
 * real cameras pass; a 65535 x 65535 frame would be ~12 GB of RGB data and
 * is clearly erroneous input.
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
 * Copy a cctv_config, safely duplicating the string pointer values.
 * The native backend ignores the path strings, but we store them verbatim
 * (they remain valid as long as the caller's config object is alive, which
 * is sufficient -- the engine does not outlive the config by contract).
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

  // Clamp thresholds to sensible range.
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
    // Log the exception in a real application; here we just return NULL.
    return nullptr;
  }
}

cctv_status cctv_engine_destroy(cctv_engine* engine) {
  std::unique_ptr<cctv_engine> owned(engine);
  return CCTV_OK;
}

cctv_status cctv_detect_objects(cctv_engine* engine, const uint8_t* rgb,
                                int width, int height, cctv_detections* out) {
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
#elif defined(CCTV_USE_DL_BACKEND)
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

  // nothrow: this is a C ABI, so an exception escaping here would be
  // undefined behaviour. The null return keeps the CCTV_ERR_ALLOC contract
  // that the malloc version had.
  out->boxes = new (std::nothrow) cctv_box[static_cast<unsigned>(out->count)];
  if (out->boxes == nullptr) {
    out->count = 0;
    return CCTV_ERR_ALLOC;
  }

  std::memcpy(out->boxes, boxes.data(),
              static_cast<unsigned>(out->count) * sizeof(cctv_box));
  return CCTV_OK;
}

cctv_status cctv_free_detections(cctv_detections* dets) {
  if (dets == nullptr) return CCTV_OK;
  delete[] dets->boxes;
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

  // Acoustic activity detection is plain code shared by every backend -- it
  // never goes through ONNX or TVM. That keeps the three build modes
  // differing only in how the image model runs, which is what makes the
  // mode comparison in bench/compare.py meaningful.
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
