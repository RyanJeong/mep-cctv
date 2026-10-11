// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file model_placeholder_test.cc
 * @brief Tests for src/backend_native/model_placeholder.cc.
 *
 * Only compiled into the placeholder configuration -- the one where the
 * TVM-generated model is absent, so this stand-in is linked in its place.
 */

#include "src/backend_native/model_placeholder.hpp"

#include <vector>

#include "gtest/gtest.h"
#include "src/backend_native/native_backend.hpp"
#include "src/core/image_ops.hpp"
#include "src/core/yolo_postprocess.hpp"

namespace {

using mep_cctv::backend_native::generated::Forward;

constexpr int kSide = mep_cctv::core::kInputSize;
constexpr int kInputElems = 3 * kSide * kSide;

/** @brief An input tensor of arbitrary but valid values. */
std::vector<float> MakeInput() {
  std::vector<float> in(static_cast<unsigned>(kInputElems));
  for (unsigned i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>(i % 255) / 255.0f;
  }
  return in;
}

TEST(ModelPlaceholderTest, ReturnsCorrectlyShapedOutput) {
  const std::vector<float> in = MakeInput();
  const std::vector<float> out = Forward(in.data(), kInputElems);
  const int expected =
      mep_cctv::core::kOutputStride * mep_cctv::core::kNumAnchors;
  EXPECT_EQ(expected, static_cast<int>(out.size()))
      << "the decode path must still see a real tensor shape";
}

// Placeholder output must not fabricate detections.
TEST(ModelPlaceholderTest, OutputIsAllZeros) {
  const std::vector<float> in = MakeInput();
  const std::vector<float> out = Forward(in.data(), kInputElems);
  for (unsigned i = 0; i < out.size(); ++i) {
    ASSERT_FLOAT_EQ(0.0f, out[i]) << "non-zero at element " << i;
  }
}

// End to end through the real backend, which is what the app calls.
TEST(ModelPlaceholderTest, BackendReportsNoDetections) {
  const std::vector<float> in = MakeInput();
  const mep_cctv::backend_native::NativeBackend backend(0.25f, 0.45f);
  mep_cctv::core::LetterboxParams params;
  params.scale = 1.0f;
  params.pad_w = 0.0f;
  params.pad_h = 0.0f;
  const std::vector<cctv_box> boxes =
      backend.Detect(in.data(), kInputElems, params, mep_cctv::core::kInputSize,
                     mep_cctv::core::kInputSize);
  EXPECT_TRUE(boxes.empty())
      << "the placeholder must never fabricate a detection";
}

// Placeholder output is independent of image content.
TEST(ModelPlaceholderTest, IsIndependentOfInput) {
  std::vector<float> a(static_cast<unsigned>(kInputElems), 0.0f);
  std::vector<float> b(static_cast<unsigned>(kInputElems), 1.0f);
  EXPECT_EQ(Forward(a.data(), kInputElems), Forward(b.data(), kInputElems));
}

}  // namespace
