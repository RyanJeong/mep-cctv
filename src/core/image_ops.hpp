// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_CORE_IMAGE_OPS_HPP_
#define SRC_CORE_IMAGE_OPS_HPP_

/**
 * @file image_ops.hpp
 * @brief Image preprocessing: letterbox-resize RGB uint8 -> float NCHW
 *        for YOLOv8 kInputSize x kInputSize input.
 *
 * Two compiled copies exist:
 *   image_ops_scalar.cc  -- portable C++14 reference
 *   image_ops_simd.cc    -- NEON (arm64) / AVX2 (x86_64) fast path
 *
 * Both copies satisfy the same contract and produce numerically equivalent
 * results (max absolute difference <= 1e-5 per element).
 */

#include <cstdint>
#include <vector>

namespace mep_cctv {
namespace core {

/**
 * @brief Side-length of the square YOLOv8 input tensor.
 *
 * Exported models must use this size. See README.md, Appendix: design
 * decisions.
 */
constexpr int kInputSize = 320;

/**
 * @brief Number of COCO object classes supported by YOLOv8.
 */
constexpr int kNumClasses = 80;

/**
 * @brief Number of elements in a single YOLOv8 input tensor (1*C*H*W).
 *
 * Layout: planar NCHW, 3 channels, kInputSize x kInputSize.
 */
constexpr int kInputElements = 3 * kInputSize * kInputSize;

/**
 * @brief Padding pixel value used by letterbox (114 / 255).
 */
constexpr float kLetterboxPad = 114.0f / 255.0f;

/**
 * @brief Result of a letterbox computation (scale and padding amounts).
 *
 * These values are required by yolo_postprocess to unmap predicted box
 * coordinates back to the original image coordinate space.
 */
struct LetterboxParams {
  float scale; /**< Uniform scale applied to the original image */
  float pad_w; /**< Total horizontal padding added (pixels, split evenly) */
  float pad_h; /**< Total vertical padding added (pixels, split evenly) */
};

/**
 * @brief Compute letterbox parameters without performing the resize.
 *
 * Pure function -- no memory allocation.
 *
 * @param src_w  Source image width  (pixels, > 0).
 * @param src_h  Source image height (pixels, > 0).
 * @return LetterboxParams describing the scale and padding that will be
 *         applied when ResizeLetterbox is called.
 *
 * @side_effects None -- referentially transparent.
 */
LetterboxParams ComputeLetterboxParams(int src_w, int src_h) noexcept;

/**
 * @brief Letterbox-resize an 8-bit RGB image into a float NCHW tensor.
 *
 * Scales the source image uniformly to fit within kInputSize x kInputSize,
 * then pads the remaining area with kLetterboxPad (114/255).
 * Output is planar NCHW, values in [0, 1].
 *
 * @param rgb     Row-major H x W x 3 uint8 source buffer.
 * @param src_w   Source width  (pixels, > 0).
 * @param src_h   Source height (pixels, > 0).
 * @param params  Letterbox parameters from ComputeLetterboxParams.
 * @return        Float vector of length kInputElements in NCHW order.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason       Allocates a heap vector proportional to the output
 *                            tensor size (kInputElements * sizeof(float)).
 * @side_effects_what         Heap allocation via std::vector constructor.
 * @side_effects_impact       kInputElements * sizeof(float) bytes per tensor.
 * @side_effects_alternatives A caller-supplied buffer would avoid this but
 *                            would force callers to know the tensor layout
 *                            size.
 */
std::vector<float> ResizeLetterbox(const uint8_t* rgb, int src_w, int src_h,
                                   const LetterboxParams& params) noexcept;

}  // namespace core
}  // namespace mep_cctv

#endif  // SRC_CORE_IMAGE_OPS_HPP_
