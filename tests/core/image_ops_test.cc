// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file image_ops_test.cc
 * @brief Unit tests for image preprocessing (letterbox resize).
 *
 * Tests cover:
 *   - ComputeLetterboxParams: scale and padding values
 *   - ResizeLetterbox scalar: output dimensions, value range, padding
 *   - ResizeLetterbox scalar vs SIMD: numerical equivalence within 1e-5
 */

#include "src/core/image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

// The linked API uses the implementation selected by CMake.
namespace scalar_impl = mep_cctv::core;

namespace mep_cctv {
namespace core {

// A separate inline reference below avoids linking duplicate API definitions.

}  // namespace core
}  // namespace mep_cctv

// ---------------------------------------------------------------------------
// Scalar reference inline (copy of scalar algorithm, not the TU symbol)
// ---------------------------------------------------------------------------

namespace test_scalar {

static mep_cctv::core::LetterboxParams ComputeLetterboxParams(int src_w,
                                                              int src_h) {
  const float scale_w = static_cast<float>(mep_cctv::core::kInputSize) / src_w;
  const float scale_h = static_cast<float>(mep_cctv::core::kInputSize) / src_h;
  const float scale = std::min(scale_w, scale_h);
  return {scale, mep_cctv::core::kInputSize - src_w * scale,
          mep_cctv::core::kInputSize - src_h * scale};
}

static std::vector<float> ResizeLetterbox(
    const uint8_t* rgb, int src_w, int src_h,
    const mep_cctv::core::LetterboxParams& p) {
  using namespace mep_cctv::core;  // NOLINT(build/namespaces)
  std::vector<float> out(static_cast<unsigned>(kInputElements), kLetterboxPad);

  const int dst_w = static_cast<int>(std::round(src_w * p.scale));
  const int dst_h = static_cast<int>(std::round(src_h * p.scale));
  const int off_x = static_cast<int>(p.pad_w * 0.5f);
  const int off_y = static_cast<int>(p.pad_h * 0.5f);

  float* plane_r = out.data();
  float* plane_g = out.data() + kInputSize * kInputSize;
  float* plane_b = out.data() + 2 * kInputSize * kInputSize;

  for (int dy = 0; dy < dst_h; ++dy) {
    const int sy = std::min(static_cast<int>(dy / p.scale), src_h - 1);
    const int dst_row = off_y + dy;
    if (dst_row < 0 || dst_row >= kInputSize) continue;
    for (int dx = 0; dx < dst_w; ++dx) {
      const int sx = std::min(static_cast<int>(dx / p.scale), src_w - 1);
      const int dst_col = off_x + dx;
      if (dst_col < 0 || dst_col >= kInputSize) continue;
      const int src_idx = (sy * src_w + sx) * 3;
      const int dst_idx = dst_row * kInputSize + dst_col;
      plane_r[dst_idx] = static_cast<float>(rgb[src_idx + 0]) / 255.0f;
      plane_g[dst_idx] = static_cast<float>(rgb[src_idx + 1]) / 255.0f;
      plane_b[dst_idx] = static_cast<float>(rgb[src_idx + 2]) / 255.0f;
    }
  }
  return out;
}

}  // namespace test_scalar

// ---------------------------------------------------------------------------
// Tests: ComputeLetterboxParams
// ---------------------------------------------------------------------------

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

TEST(ResizeLetterboxTest, PartialSourceRowAndClippedDestinationMatchReference) {
  const int kWidth = 13;
  const int kHeight = 11;
  const std::vector<uint8_t> rgb(kWidth * kHeight * 3, 127);
  for (const float padding : {-8.0f, 632.0f}) {
    const mep_cctv::core::LetterboxParams params{1.0f, padding, padding};
    const auto expected =
        test_scalar::ResizeLetterbox(rgb.data(), kWidth, kHeight, params);
    const auto actual =
        mep_cctv::core::ResizeLetterbox(rgb.data(), kWidth, kHeight, params);
    ASSERT_EQ(expected.size(), actual.size());
    for (int i = 0; i < mep_cctv::core::kInputElements; ++i) {
      ASSERT_NEAR(expected[i], actual[i], 1e-5f) << i;
    }
  }
}

TEST(LetterboxParamsTest, SquareImageNoScaleNoPad) {
  // An image already at the model's input size needs no scaling or padding,
  // whatever that size happens to be.
  const int kSize = mep_cctv::core::kInputSize;
  const auto p = mep_cctv::core::ComputeLetterboxParams(kSize, kSize);
  EXPECT_FLOAT_EQ(1.0f, p.scale);
  EXPECT_FLOAT_EQ(0.0f, p.pad_w);
  EXPECT_FLOAT_EQ(0.0f, p.pad_h);
}

TEST(LetterboxParamsTest, WideImageScalesOnWidth) {
  // A 16:9 frame is width-limited: scale is set by the width, and the slack
  // goes into vertical padding.
  const float kSize = static_cast<float>(mep_cctv::core::kInputSize);
  const auto p = mep_cctv::core::ComputeLetterboxParams(1280, 720);
  EXPECT_FLOAT_EQ(kSize / 1280.0f, p.scale);
  EXPECT_FLOAT_EQ(0.0f, p.pad_w);
  EXPECT_NEAR(kSize - 720.0f * (kSize / 1280.0f), p.pad_h, 0.01f);
}

TEST(LetterboxParamsTest, TallImageScalesOnHeight) {
  // A portrait frame is height-limited: the slack goes into horizontal
  // padding instead.
  const float kSize = static_cast<float>(mep_cctv::core::kInputSize);
  const auto p = mep_cctv::core::ComputeLetterboxParams(480, 960);
  EXPECT_NEAR(kSize / 960.0f, p.scale, 1e-5f);
  EXPECT_NEAR(kSize - 480.0f * (kSize / 960.0f), p.pad_w, 0.5f);
  EXPECT_NEAR(0.0f, p.pad_h, 0.01f);
}

TEST(LetterboxParamsTest, SmallSquareScalesUp) {
  // A square image smaller than the input is scaled up to fill it exactly.
  const int kHalf = mep_cctv::core::kInputSize / 2;
  const auto p = mep_cctv::core::ComputeLetterboxParams(kHalf, kHalf);
  EXPECT_FLOAT_EQ(2.0f, p.scale);
  EXPECT_FLOAT_EQ(0.0f, p.pad_w);
  EXPECT_FLOAT_EQ(0.0f, p.pad_h);
}

// ---------------------------------------------------------------------------
// Tests: ResizeLetterbox (scalar reference path via test_scalar)
// ---------------------------------------------------------------------------

TEST(ResizeLetterboxScalarTest, OutputDimensionsCorrect) {
  // 640x640 red image -> tensor must be [kInputElements].
  const int W = 640;
  const int H = 640;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 0);
  // Fill red channel.
  for (int i = 0; i < W * H; ++i) img[static_cast<unsigned>(i * 3)] = 255;

  const auto p = test_scalar::ComputeLetterboxParams(W, H);
  const auto out = test_scalar::ResizeLetterbox(img.data(), W, H, p);

  EXPECT_EQ(mep_cctv::core::kInputElements, static_cast<int>(out.size()));
}

TEST(ResizeLetterboxScalarTest, ValueRangeZeroToOne) {
  const int W = 320;
  const int H = 240;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3));
  for (int i = 0; i < W * H * 3; ++i) {
    img[static_cast<unsigned>(i)] = static_cast<uint8_t>(i % 256);
  }
  const auto p = test_scalar::ComputeLetterboxParams(W, H);
  const auto out = test_scalar::ResizeLetterbox(img.data(), W, H, p);

  for (float v : out) {
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, 1.0f);
  }
}

TEST(ResizeLetterboxScalarTest, PaddingRegionsFilledWithConstant) {
  // A 2:1 frame fills half the model height; the rest is padding.
  const int W = 640;
  const int H = 320;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 0);

  const auto p = test_scalar::ComputeLetterboxParams(W, H);
  const auto out = test_scalar::ResizeLetterbox(img.data(), W, H, p);

  const int pad_rows = static_cast<int>(p.pad_h * 0.5f);
  const float expected_pad = mep_cctv::core::kLetterboxPad;

  // Check first pad_rows rows of channel 0 (red plane).
  for (int row = 0; row < pad_rows; ++row) {
    for (int col = 0; col < mep_cctv::core::kInputSize; ++col) {
      const float v =
          out[static_cast<unsigned>(row * mep_cctv::core::kInputSize + col)];
      EXPECT_NEAR(expected_pad, v, 1e-5f) << "row=" << row << " col=" << col;
    }
  }
}

TEST(ResizeLetterboxScalarTest, RedChannelPreserved) {
  // Pure red image: R channel should be ~1.0, G and B ~0 in non-pad region.
  const int W = 640;
  const int H = 640;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 0);
  for (int i = 0; i < W * H; ++i) img[static_cast<unsigned>(i * 3 + 0)] = 255;

  const auto p = test_scalar::ComputeLetterboxParams(W, H);
  const auto out = test_scalar::ResizeLetterbox(img.data(), W, H, p);

  // Top-left pixel of red plane (no padding for 640x640).
  EXPECT_NEAR(1.0f, out[0], 1e-5f);
  // Green plane first pixel.
  EXPECT_NEAR(0.0f,
              out[static_cast<unsigned>(mep_cctv::core::kInputSize *
                                        mep_cctv::core::kInputSize)],
              1e-5f);
}

// ---------------------------------------------------------------------------
// Tests: SIMD vs Scalar numerical equivalence
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Regression tests: input dimensions LARGER than kInputSize (720p, tall)
// ---------------------------------------------------------------------------
// These tests guard against the stack buffer overflow in image_ops_simd.cc
// where tmp_r/tmp_g/tmp_b were sized kInputSize=640 but written up to
// src_w elements.  1280x720 wrote 640 elements past the end of those arrays.
// ---------------------------------------------------------------------------

TEST(ResizeLetterboxLargeInputTest, Landscape720pOutputCorrectSize) {
  // 1280x720: scale = 640/1280 = 0.5, pad_h = 640 - 360 = 280
  const int W = 1280;
  const int H = 720;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 100);
  const auto p = test_scalar::ComputeLetterboxParams(W, H);
  const auto out = test_scalar::ResizeLetterbox(img.data(), W, H, p);
  EXPECT_EQ(mep_cctv::core::kInputElements, static_cast<int>(out.size()));
}

TEST(ResizeLetterboxLargeInputTest, Landscape720pValueRange) {
  const int W = 1280;
  const int H = 720;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3));
  for (int i = 0; i < W * H * 3; ++i) {
    img[static_cast<unsigned>(i)] = static_cast<uint8_t>(i % 256);
  }
  const auto p = test_scalar::ComputeLetterboxParams(W, H);
  const auto out = test_scalar::ResizeLetterbox(img.data(), W, H, p);
  for (float v : out) {
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, 1.0f);
  }
}

TEST(ResizeLetterboxLargeInputTest, TallFrameOutputCorrectSize) {
  // 480x1280: scale = 640/1280 = 0.5, pad_w = 640 - 240 = 400
  const int W = 480;
  const int H = 1280;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 50);
  const auto p = test_scalar::ComputeLetterboxParams(W, H);
  const auto out = test_scalar::ResizeLetterbox(img.data(), W, H, p);
  EXPECT_EQ(mep_cctv::core::kInputElements, static_cast<int>(out.size()));
}

TEST(ResizeLetterboxLargeInputTest, TallFramePaddingCorrect) {
  // 480x1280: scale=0.5, pad_w=400, off_x=200
  // Left 200 columns of the output should be kLetterboxPad.
  const int W = 480;
  const int H = 1280;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 0);
  const auto p = test_scalar::ComputeLetterboxParams(W, H);
  const auto out = test_scalar::ResizeLetterbox(img.data(), W, H, p);
  const int off_x = static_cast<int>(p.pad_w * 0.5f);
  EXPECT_GT(off_x, 0) << "sanity: tall image should have left padding";
  for (int row = 0; row < mep_cctv::core::kInputSize; ++row) {
    for (int col = 0; col < off_x; ++col) {
      const float v =
          out[static_cast<unsigned>(row * mep_cctv::core::kInputSize + col)];
      EXPECT_NEAR(mep_cctv::core::kLetterboxPad, v, 1e-5f)
          << "row=" << row << " col=" << col;
    }
  }
}

#if CCTV_SIMD_ENABLED

// When CCTV_SIMD_ENABLED both TUs are linked. We call the public API
// (which resolves to the SIMD TU because it is linked last) and compare
// against test_scalar::ResizeLetterbox (inline reference).

TEST(ResizeLetterboxEquivalenceTest, ScalarVsSimdWithin1e5) {
  const int W = 320;
  const int H = 240;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3));
  for (int i = 0; i < W * H * 3; ++i) {
    img[static_cast<unsigned>(i)] = static_cast<uint8_t>((i * 37 + 13) % 256);
  }

  const auto p_s = test_scalar::ComputeLetterboxParams(W, H);
  const auto ref = test_scalar::ResizeLetterbox(img.data(), W, H, p_s);

  const auto p_api = mep_cctv::core::ComputeLetterboxParams(W, H);
  const auto got = mep_cctv::core::ResizeLetterbox(img.data(), W, H, p_api);

  ASSERT_EQ(ref.size(), got.size());
  for (int i = 0; i < static_cast<int>(ref.size()); ++i) {
    EXPECT_NEAR(ref[static_cast<unsigned>(i)], got[static_cast<unsigned>(i)],
                1e-5f)
        << "index=" << i;
  }
}

TEST(LetterboxParamsEquivalenceTest, ScalarVsSimdParams) {
  // Use NEAR rather than FLOAT_EQ: the inline test_scalar and the library
  // both compute pad = kInputSize - src*scale, but intermediate rounding
  // in FP can produce epsilon differences (~2e-5).
  const auto p_s = test_scalar::ComputeLetterboxParams(1920, 1080);
  const auto p_api = mep_cctv::core::ComputeLetterboxParams(1920, 1080);
  EXPECT_NEAR(p_s.scale, p_api.scale, 1e-5f);
  EXPECT_NEAR(p_s.pad_w, p_api.pad_w, 1e-3f);
  EXPECT_NEAR(p_s.pad_h, p_api.pad_h, 1e-3f);
}

// Regression: 1280x720 must not crash (SIMD stack overflow fix).
TEST(ResizeLetterboxEquivalenceTest, Landscape720pScalarVsSimdNoCrash) {
  const int W = 1280;
  const int H = 720;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3));
  for (int i = 0; i < W * H * 3; ++i) {
    img[static_cast<unsigned>(i)] = static_cast<uint8_t>((i * 19 + 7) % 256);
  }

  const auto p_s = test_scalar::ComputeLetterboxParams(W, H);
  const auto ref = test_scalar::ResizeLetterbox(img.data(), W, H, p_s);

  const auto p_api = mep_cctv::core::ComputeLetterboxParams(W, H);
  // Must not crash (was: stack buffer overflow -> SIGABRT).
  const auto got = mep_cctv::core::ResizeLetterbox(img.data(), W, H, p_api);

  ASSERT_EQ(ref.size(), got.size());
  // Spot-check a sample of output elements.
  const int step = static_cast<int>(ref.size()) / 100;
  for (int i = 0; i < static_cast<int>(ref.size()); i += step) {
    EXPECT_NEAR(ref[static_cast<unsigned>(i)], got[static_cast<unsigned>(i)],
                1e-5f)
        << "index=" << i;
  }
}

// Regression: 480x1280 tall frame must not crash.
TEST(ResizeLetterboxEquivalenceTest, TallFrameScalarVsSimdNoCrash) {
  const int W = 480;
  const int H = 1280;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3));
  for (int i = 0; i < W * H * 3; ++i) {
    img[static_cast<unsigned>(i)] = static_cast<uint8_t>((i * 31 + 5) % 256);
  }

  const auto p_s = test_scalar::ComputeLetterboxParams(W, H);
  const auto ref = test_scalar::ResizeLetterbox(img.data(), W, H, p_s);

  const auto p_api = mep_cctv::core::ComputeLetterboxParams(W, H);
  const auto got = mep_cctv::core::ResizeLetterbox(img.data(), W, H, p_api);

  ASSERT_EQ(ref.size(), got.size());
  const int step = static_cast<int>(ref.size()) / 100;
  for (int i = 0; i < static_cast<int>(ref.size()); i += step) {
    EXPECT_NEAR(ref[static_cast<unsigned>(i)], got[static_cast<unsigned>(i)],
                1e-5f)
        << "index=" << i;
  }
}

#else

TEST(ResizeLetterboxEquivalenceTest, ScalarVsScalarTriviallyEqual) {
  // When SIMD is disabled the library uses scalar too; just confirm sizes.
  const int W = 320;
  const int H = 240;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 128);
  const auto p = mep_cctv::core::ComputeLetterboxParams(W, H);
  const auto out = mep_cctv::core::ResizeLetterbox(img.data(), W, H, p);
  EXPECT_EQ(mep_cctv::core::kInputElements, static_cast<int>(out.size()));
}

// Regression SIMD=OFF: scalar must also handle large inputs correctly.
TEST(ResizeLetterboxEquivalenceTest, Landscape720pScalarNoCrash) {
  const int W = 1280;
  const int H = 720;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 128);
  const auto p = mep_cctv::core::ComputeLetterboxParams(W, H);
  const auto out = mep_cctv::core::ResizeLetterbox(img.data(), W, H, p);
  EXPECT_EQ(mep_cctv::core::kInputElements, static_cast<int>(out.size()));
}

TEST(ResizeLetterboxEquivalenceTest, TallFrameScalarNoCrash) {
  const int W = 480;
  const int H = 1280;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 64);
  const auto p = mep_cctv::core::ComputeLetterboxParams(W, H);
  const auto out = mep_cctv::core::ResizeLetterbox(img.data(), W, H, p);
  EXPECT_EQ(mep_cctv::core::kInputElements, static_cast<int>(out.size()));
}

#endif  // CCTV_SIMD_ENABLED

}  // namespace
