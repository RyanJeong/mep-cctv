// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file image_ops_scalar.cc
 * @brief Scalar (portable C++14) implementation of image_ops.
 *
 * This file is compiled for onnx mode, CCTV_SCALAR_KERNELS=ON, or architectures
 * where no SIMD path is provided.  It is also used as the reference
 * implementation for numerical-equivalence tests against the SIMD copy.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "src/core/image_ops.hpp"

namespace mep_cctv {
namespace core {

LetterboxParams ComputeLetterboxParams(int src_w, int src_h) noexcept {
  assert(src_w > 0 && src_h > 0);

  const float scale_w =
      static_cast<float>(kInputSize) / static_cast<float>(src_w);
  const float scale_h =
      static_cast<float>(kInputSize) / static_cast<float>(src_h);
  const float scale = std::min(scale_w, scale_h);
  const float new_w = static_cast<float>(src_w) * scale;
  const float new_h = static_cast<float>(src_h) * scale;
  const float pad_w = kInputSize - new_w;
  const float pad_h = kInputSize - new_h;
  return LetterboxParams{scale, pad_w, pad_h};
}

std::vector<float> ResizeLetterbox(const uint8_t* rgb, int src_w, int src_h,
                                   const LetterboxParams& params) noexcept {
  assert(rgb != nullptr && src_w > 0 && src_h > 0);

  std::vector<float> out(kInputElements, kLetterboxPad);

  // Destination dimensions of the scaled image (before padding).
  const auto dst_w =
      static_cast<int>(std::round(static_cast<float>(src_w) * params.scale));
  const auto dst_h =
      static_cast<int>(std::round(static_cast<float>(src_h) * params.scale));

  // Top-left offset where the scaled image is placed in the padded canvas.
  const auto off_x = static_cast<int>(params.pad_w * 0.5f);
  const auto off_y = static_cast<int>(params.pad_h * 0.5f);

  // Pointer bases for each output plane (planar NCHW).
  float* plane_r = out.data();
  float* plane_g = out.data() + kInputSize * kInputSize;
  float* plane_b = out.data() + 2 * kInputSize * kInputSize;

  for (int dy = 0; dy < dst_h; ++dy) {
    // Nearest-neighbour source row index.
    const int sy = std::min(
        static_cast<int>(static_cast<float>(dy) / params.scale), src_h - 1);
    const int dst_row = off_y + dy;
    if (dst_row < 0 || dst_row >= kInputSize) continue;

    for (int dx = 0; dx < dst_w; ++dx) {
      const int sx = std::min(
          static_cast<int>(static_cast<float>(dx) / params.scale), src_w - 1);
      const int dst_col = off_x + dx;
      if (dst_col < 0 || dst_col >= kInputSize) continue;

      // Full-resolution RGB offsets can exceed the range of int.
      const int64_t src_idx = (static_cast<int64_t>(sy) * src_w + sx) * 3;
      const int dst_idx = dst_row * kInputSize + dst_col;

      plane_r[dst_idx] = static_cast<float>(rgb[src_idx + 0]) / 255.0f;
      plane_g[dst_idx] = static_cast<float>(rgb[src_idx + 1]) / 255.0f;
      plane_b[dst_idx] = static_cast<float>(rgb[src_idx + 2]) / 255.0f;
    }
  }

  return out;
}

}  // namespace core
}  // namespace mep_cctv
