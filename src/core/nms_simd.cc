// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file nms_simd.cc
 * @brief SIMD-accelerated NMS implementation.
 *
 * The SIMD opportunity in NMS is the inner IoU loop: for a chosen "keeper"
 * box we compute IoU against all remaining candidates.  On arm64 NEON we
 * process 4 candidates per iteration; on x86_64 AVX2 we process 8.
 *
 * ComputeIoU (single-pair) is identical to the scalar version.
 * ApplyNMS reuses the same greedy algorithm with a vectorised inner loop.
 *
 * CMake selects this file when scalar kernels are disabled.
 */

#include <algorithm>
#include <array>
#include <vector>

#include "src/core/nms.hpp"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CCTV_NMS_HAVE_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define CCTV_NMS_HAVE_AVX2 1
#endif

namespace mep_cctv {
namespace core {

// ComputeIoU is scalar -- called for individual pairs (e.g. unit tests).
float ComputeIoU(const cctv_box& a, const cctv_box& b) noexcept {
  const float ax2 = a.x + a.w;
  const float ay2 = a.y + a.h;
  const float bx2 = b.x + b.w;
  const float by2 = b.y + b.h;

  const float inter_x1 = std::max(a.x, b.x);
  const float inter_y1 = std::max(a.y, b.y);
  const float inter_x2 = std::min(ax2, bx2);
  const float inter_y2 = std::min(ay2, by2);

  const float inter_w = inter_x2 - inter_x1;
  const float inter_h = inter_y2 - inter_y1;
  if (inter_w <= 0.0f || inter_h <= 0.0f) return 0.0f;

  const float inter_area = inter_w * inter_h;
  const float area_a = a.w * a.h;
  const float area_b = b.w * b.h;
  const float union_area = area_a + area_b - inter_area;
  if (union_area <= 0.0f) return 0.0f;

  return inter_area / union_area;
}

// ---------------------------------------------------------------------------
// SIMD helpers for the inner suppression loop
// ---------------------------------------------------------------------------

#if defined(CCTV_NMS_HAVE_NEON)

// Returns a bitmask (4 bits) indicating which of 4 candidates have
// IoU > threshold with the keeper box.
static uint32_t IoUMask4(const cctv_box& keeper, const cctv_box* cands,
                         float threshold) noexcept {
  // Load keeper coordinates.
  const float32x4_t kx1 = vdupq_n_f32(keeper.x);
  const float32x4_t ky1 = vdupq_n_f32(keeper.y);
  const float32x4_t kx2 = vdupq_n_f32(keeper.x + keeper.w);
  const float32x4_t ky2 = vdupq_n_f32(keeper.y + keeper.h);
  const float32x4_t ka = vdupq_n_f32(keeper.w * keeper.h);
  const float32x4_t thresh = vdupq_n_f32(threshold);
  const float32x4_t zero = vdupq_n_f32(0.0f);

  // Gather candidate coordinates into contiguous SIMD lanes.
  std::array<float, 4> cx1_arr;
  std::array<float, 4> cy1_arr;
  std::array<float, 4> cx2_arr;
  std::array<float, 4> cy2_arr;
  std::array<float, 4> ca_arr;
  for (int i = 0; i < 4; ++i) {
    cx1_arr[i] = cands[i].x;
    cy1_arr[i] = cands[i].y;
    cx2_arr[i] = cands[i].x + cands[i].w;
    cy2_arr[i] = cands[i].y + cands[i].h;
    ca_arr[i] = cands[i].w * cands[i].h;
  }
  const float32x4_t cx1 = vld1q_f32(cx1_arr.data());
  const float32x4_t cy1 = vld1q_f32(cy1_arr.data());
  const float32x4_t cx2 = vld1q_f32(cx2_arr.data());
  const float32x4_t cy2 = vld1q_f32(cy2_arr.data());
  const float32x4_t ca = vld1q_f32(ca_arr.data());

  // Intersection.
  const float32x4_t ix1 = vmaxq_f32(kx1, cx1);
  const float32x4_t iy1 = vmaxq_f32(ky1, cy1);
  const float32x4_t ix2 = vminq_f32(kx2, cx2);
  const float32x4_t iy2 = vminq_f32(ky2, cy2);
  const float32x4_t iw = vmaxq_f32(vsubq_f32(ix2, ix1), zero);
  const float32x4_t ih = vmaxq_f32(vsubq_f32(iy2, iy1), zero);
  const float32x4_t inter = vmulq_f32(iw, ih);

  // Union.
  const float32x4_t uni = vsubq_f32(vaddq_f32(ka, ca), inter);

  // Approximate inter / union with a reciprocal estimate and Newton refinement.
  const float32x4_t rcp_uni = vrecpeq_f32(uni);
  const float32x4_t rcp1 = vmulq_f32(vrecpsq_f32(uni, rcp_uni), rcp_uni);
  const float32x4_t iou = vmulq_f32(inter, rcp1);

  // Mask: iou > threshold
  const uint32x4_t mask = vcgtq_f32(iou, thresh);
  std::array<uint32_t, 4> result;
  vst1q_u32(result.data(), mask);
  // Each lane is 0xFFFFFFFF if suppressed, 0 otherwise.
  return (result[0] ? 1u : 0u) | (result[1] ? 2u : 0u) | (result[2] ? 4u : 0u) |
         (result[3] ? 8u : 0u);
}

#elif defined(CCTV_NMS_HAVE_AVX2)

// Returns a bitmask (8 bits) indicating which of 8 candidates have
// IoU > threshold with the keeper box.
static uint32_t IoUMask8(const cctv_box& keeper, const cctv_box* cands,
                         float threshold) noexcept {
  const __m256 kx1 = _mm256_set1_ps(keeper.x);
  const __m256 ky1 = _mm256_set1_ps(keeper.y);
  const __m256 kx2 = _mm256_set1_ps(keeper.x + keeper.w);
  const __m256 ky2 = _mm256_set1_ps(keeper.y + keeper.h);
  const __m256 ka = _mm256_set1_ps(keeper.w * keeper.h);
  const __m256 thresh = _mm256_set1_ps(threshold);
  const __m256 zero = _mm256_setzero_ps();

  std::array<float, 8> cx1_arr;
  std::array<float, 8> cy1_arr;
  std::array<float, 8> cx2_arr;
  std::array<float, 8> cy2_arr;
  std::array<float, 8> ca_arr;
  for (int i = 0; i < 8; ++i) {
    cx1_arr[i] = cands[i].x;
    cy1_arr[i] = cands[i].y;
    cx2_arr[i] = cands[i].x + cands[i].w;
    cy2_arr[i] = cands[i].y + cands[i].h;
    ca_arr[i] = cands[i].w * cands[i].h;
  }
  const __m256 cx1 = _mm256_loadu_ps(cx1_arr.data());
  const __m256 cy1 = _mm256_loadu_ps(cy1_arr.data());
  const __m256 cx2 = _mm256_loadu_ps(cx2_arr.data());
  const __m256 cy2 = _mm256_loadu_ps(cy2_arr.data());
  const __m256 ca = _mm256_loadu_ps(ca_arr.data());

  const __m256 ix1 = _mm256_max_ps(kx1, cx1);
  const __m256 iy1 = _mm256_max_ps(ky1, cy1);
  const __m256 ix2 = _mm256_min_ps(kx2, cx2);
  const __m256 iy2 = _mm256_min_ps(ky2, cy2);
  const __m256 iw = _mm256_max_ps(_mm256_sub_ps(ix2, ix1), zero);
  const __m256 ih = _mm256_max_ps(_mm256_sub_ps(iy2, iy1), zero);
  const __m256 inter = _mm256_mul_ps(iw, ih);
  const __m256 uni = _mm256_sub_ps(_mm256_add_ps(ka, ca), inter);

  // iou = inter / union
  const __m256 iou = _mm256_div_ps(inter, uni);
  // mask: iou > threshold
  const __m256 mask = _mm256_cmp_ps(iou, thresh, _CMP_GT_OQ);
  return static_cast<uint32_t>(_mm256_movemask_ps(mask));
}

#endif  // CCTV_NMS_HAVE_NEON / CCTV_NMS_HAVE_AVX2

#if defined(CCTV_NMS_HAVE_NEON) || defined(CCTV_NMS_HAVE_AVX2)
/**
 * @brief Mark every lane the suppression mask has set.
 *
 * @param mask       Bit i set means candidate base+i is suppressed.
 * @param base       Index of lane 0 within suppressed.
 * @param lanes      Number of lanes the mask describes.
 * @param suppressed Flags to update; must hold at least base+lanes entries.
 * @return Number of set mask lanes written, including already-suppressed lanes.
 *
 * @side_effects VIOLATION: Non-referentially transparent
 * @side_effects_reason Updates the caller's suppression flags in place.
 * @side_effects_what   Writes up to `lanes` bools into *suppressed.
 * @side_effects_impact O(lanes).
 * @side_effects_alternatives Returning a new vector would allocate once per
 *                      block of candidates, inside the hot loop of NMS.
 */
int SuppressMaskedLanes(uint32_t mask, int base, int lanes,
                        std::vector<bool>* suppressed) {
  int marked = 0;
  for (int k = 0; k < lanes; ++k) {
    if ((mask & (1u << k)) != 0u) {
      (*suppressed)[static_cast<unsigned>(base + k)] = true;
      ++marked;
    }
  }
  return marked;
}
#endif  // CCTV_NMS_HAVE_NEON || CCTV_NMS_HAVE_AVX2

// ---------------------------------------------------------------------------
// ApplyNMS -- greedy algorithm with optional SIMD inner loop
// ---------------------------------------------------------------------------

std::vector<cctv_box> ApplyNMS(const std::vector<cctv_box>& boxes,
                               float iou_threshold) noexcept {
  if (boxes.empty()) return {};

  std::vector<cctv_box> sorted = boxes;
  std::sort(
      sorted.begin(), sorted.end(),
      [](const cctv_box& a, const cctv_box& b) { return a.score > b.score; });

  const auto n = static_cast<int>(sorted.size());
  std::vector<bool> suppressed(static_cast<unsigned>(n), false);
  std::vector<cctv_box> result;

  for (int i = 0; i < n; ++i) {
    if (suppressed[static_cast<unsigned>(i)]) continue;
    result.push_back(sorted[static_cast<unsigned>(i)]);

    const cctv_box& keeper = sorted[static_cast<unsigned>(i)];
    int j = i + 1;

#if defined(CCTV_NMS_HAVE_NEON)
    // Process 4 candidates at a time.
    for (; j + 4 <= n; j += 4) {
      if (suppressed[static_cast<unsigned>(j)] &&
          suppressed[static_cast<unsigned>(j + 1)] &&
          suppressed[static_cast<unsigned>(j + 2)] &&
          suppressed[static_cast<unsigned>(j + 3)]) {
        continue;
      }
      const uint32_t mask = IoUMask4(keeper, sorted.data() + j, iou_threshold);
      SuppressMaskedLanes(mask, j, 4, &suppressed);
    }
#elif defined(CCTV_NMS_HAVE_AVX2)
    // Process 8 candidates at a time.
    for (; j + 8 <= n; j += 8) {
      const uint32_t mask = IoUMask8(keeper, sorted.data() + j, iou_threshold);
      SuppressMaskedLanes(mask, j, 8, &suppressed);
    }
#endif

    // Scalar tail.
    for (; j < n; ++j) {
      if (suppressed[static_cast<unsigned>(j)]) continue;
      if (ComputeIoU(keeper, sorted[static_cast<unsigned>(j)]) >
          iou_threshold) {
        suppressed[static_cast<unsigned>(j)] = true;
      }
    }
  }

  return result;
}

}  // namespace core
}  // namespace mep_cctv
