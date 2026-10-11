// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef CCTV_CCTV_H_
#define CCTV_CCTV_H_

/**
 * @file cctv.h
 * @brief Public C-ABI for the mep-cctv inference engine.
 *
 * This header is shared by the native and onnx modes.
 * It is C and C++ compatible.  All allocation that crosses the ABI boundary
 * allocates the boxes array, which callers release with cctv_free_detections;
 * the allocator is an internal detail and must not be called directly.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Visibility macro
 * -----------------------------------------------------------------------*/

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef CCTV_BUILDING_SHARED
#define CCTV_API __declspec(dllexport)
#else
#define CCTV_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define CCTV_API __attribute__((visibility("default")))
#else
#define CCTV_API
#endif

/* -------------------------------------------------------------------------
 * Status codes
 * -----------------------------------------------------------------------*/

typedef enum {
  CCTV_OK = 0,          /**< Success */
  CCTV_ERR_INVALID_ARG, /**< NULL pointer or out-of-range parameter */
  CCTV_ERR_BACKEND,     /**< Backend inference failure */
  CCTV_ERR_ALLOC,       /**< Memory allocation failure */
  CCTV_ERR_UNSUPPORTED  /**< Feature not compiled in */
} cctv_status;

/* -------------------------------------------------------------------------
 * Data types
 * -----------------------------------------------------------------------*/

/**
 * @brief A single detected bounding box in the ORIGINAL image coordinate
 *        space (before any letterbox scaling).
 *
 * x, y are the top-left corner in pixels; w, h are pixel dimensions.
 */
typedef struct {
  float x;      /**< Top-left x, pixels, original image space */
  float y;      /**< Top-left y, pixels, original image space */
  float w;      /**< Box width  in pixels, original image space */
  float h;      /**< Box height in pixels, original image space */
  float score;  /**< Confidence score in [0, 1] */
  int class_id; /**< COCO class index in [0, 79] */
} cctv_box;

/** @brief Collection of detections; ownership transferred to caller. */
typedef struct {
  cctv_box* boxes; /**< Heap-allocated array; free with cctv_free_detections */
  int count;       /**< Number of valid entries in boxes */
} cctv_detections;

/** @brief Result of an acoustic activity pass. */
typedef struct {
  int person_present; /**< Non-zero when sound was detected nearby */
  float score;        /**< Short-term RMS level in [0, 1] */
} cctv_audio_result;

/** @brief Opaque engine handle; created by cctv_engine_create. */
typedef struct cctv_engine cctv_engine;

/**
 * @brief Engine configuration.
 *
 * @p yolo_model_path is used only by onnx mode and is ignored (may
 * be NULL) for native mode.
 *
 * There is no audio model path: acoustic detection is a plain algorithm (see
 * cctv_detect_audio), not a network, so it needs no weights.
 */
typedef struct {
  const char* yolo_model_path; /**< Path to YOLO ONNX model (onnx only) */
  float conf_threshold;        /**< Minimum detection confidence (0..1) */
  float iou_threshold;         /**< IoU threshold for NMS (0..1) */
} cctv_config;

/* -------------------------------------------------------------------------
 * Engine lifecycle
 * -----------------------------------------------------------------------*/

/**
 * @brief Create and initialise an inference engine.
 *
 * @param cfg  Pointer to a filled cctv_config; must not be NULL.
 * @return     Non-NULL engine handle on success; NULL on failure.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Allocates heap memory for the engine struct and
 *                            backend objects; loads model weights from disk for
 *                            onnx mode.
 * @side_effects_what         Heap allocation; file I/O for the onnx mode.
 * @side_effects_impact       O(model-size) startup cost; subsequent calls are
 *                            stateless relative to image data.
 * @side_effects_alternatives A pure factory returning a value type would
 *                            require copying large model weights on every call;
 *                            a C ABI cannot return non-trivial objects by
 *                            value.
 */
CCTV_API cctv_engine* cctv_engine_create(const cctv_config* cfg);

/**
 * @brief Destroy an engine and free all associated resources.
 *
 * @param engine  Handle returned by cctv_engine_create.  NULL is accepted
 *                (returns CCTV_OK immediately).
 * @return CCTV_OK on success.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Frees heap memory; invalidates the pointer.
 * @side_effects_what         Heap deallocation (free).
 * @side_effects_impact       O(1); after return @p engine must not be used.
 * @side_effects_alternatives None -- deallocation is inherently effectful.
 */
CCTV_API cctv_status cctv_engine_destroy(cctv_engine* engine);

/* -------------------------------------------------------------------------
 * Inference
 * -----------------------------------------------------------------------*/

/**
 * @brief Run object detection on an 8-bit RGB image.
 *
 * @param engine  Valid engine handle.
 * @param rgb     Row-major H×W×3 uint8 buffer (no padding between rows).
 * @param width   Image width in pixels (must be > 0).
 * @param height  Image height in pixels (must be > 0).
 * @param out     Caller-allocated result struct; populated on CCTV_OK.
 *                Free with cctv_free_detections when done.
 * @return CCTV_OK on success; CCTV_ERR_INVALID_ARG if any pointer is NULL
 *         or dims are non-positive; CCTV_ERR_ALLOC if allocation fails;
 *         CCTV_ERR_BACKEND on inference failure.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Allocates the cctv_detections::boxes array on the
 *                            heap; caller must call cctv_free_detections.
 * @side_effects_what         Heap allocation for the boxes array.
 * @side_effects_impact       Proportional to number of detections;
 *                            typically < 1 KB.
 * @side_effects_alternatives A caller-supplied buffer would require the
 *                            caller to know the maximum detection count a
 *                            priori, coupling caller code to internal limits.
 */
CCTV_API cctv_status cctv_detect_objects(cctv_engine* engine,
                                         const uint8_t* rgb, int width,
                                         int height, cctv_detections* out);

/**
 * @brief Free a cctv_detections struct returned by cctv_detect_objects.
 *
 * @param dets  Pointer to the struct to free.  NULL is accepted (no-op).
 * @return CCTV_OK.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Frees the heap-allocated boxes array.
 * @side_effects_what         Heap deallocation (free).
 * @side_effects_impact       O(1); after return dets->boxes is NULL and
 *                            dets->count is 0.
 * @side_effects_alternatives None -- deallocation is inherently effectful.
 */
CCTV_API cctv_status cctv_free_detections(cctv_detections* dets);

/**
 * @brief Detect acoustic activity ("is something making noise nearby?").
 *
 * This is a short-term energy (RMS) test in plain C++, NOT a classifier: it
 * reports that sound is present, not who or what made it.  Identical in every
 * mode; it is independent of object-model execution.
 *
 * @param engine       Valid engine handle.
 * @param samples      Mono PCM float array, values in [-1.0, 1.0].
 * @param num_samples  Number of samples (must be > 0).
 * @param sample_rate  Samples per second (e.g. 16000); must be > 0.
 * @param out          Caller-allocated result struct; populated on CCTV_OK.
 * @return CCTV_OK on success; CCTV_ERR_INVALID_ARG if any pointer is NULL
 *         or parameters are non-positive.
 *
 * @side_effects None -- referentially transparent given fixed engine state.
 */
/* engine is const: acoustic detection is plain code shared by both modes and
 * reads nothing from the engine. Object detection cannot make the same
 * promise -- ONNX Runtime mutates its session's internal arena -- so
 * cctv_detect_objects keeps a mutable pointer. */
CCTV_API cctv_status cctv_detect_audio(const cctv_engine* engine,
                                       const float* samples, int num_samples,
                                       int sample_rate, cctv_audio_result* out);

/* -------------------------------------------------------------------------
 * Metadata queries
 * -----------------------------------------------------------------------*/

/**
 * @brief Return the project version string (e.g. "0.1.0").
 *
 * @return Null-terminated string with static storage duration.
 *
 * @side_effects None -- referentially transparent.
 */
CCTV_API const char* cctv_version(void);

/**
 * @brief Return the compiled backend name.
 *
 * @return "native-tvm" or "dl-onnx"; static storage duration.
 *
 * @side_effects None -- referentially transparent.
 */
CCTV_API const char* cctv_backend(void);

/**
 * @brief Report whether SIMD kernels are compiled in.
 *
 * @return 1 if SIMD copies were compiled; 0 otherwise.
 *
 * @side_effects None -- referentially transparent.
 */
CCTV_API int cctv_simd_enabled(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // CCTV_CCTV_H_
