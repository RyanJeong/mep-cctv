// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file yolo_postprocess_test.cc
 * @brief Unit tests for DecodeYoloOutput.
 *
 * Tests cover:
 *   - Zero tensor -> no detections
 *   - Synthetic tensor -> expected detections
 *   - Confidence threshold filtering
 *   - NMS applied correctly
 *   - Coordinate un-letterboxing
 *   - Edge cases: negative coords clamped, large boxes clamped
 */

#include "src/core/yolo_postprocess.hpp"

#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "src/core/image_ops.hpp"

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Helper: create an all-zero output tensor.
static std::vector<float> ZeroTensor() {
  return std::vector<float>(
      static_cast<unsigned>(mep_cctv::core::kOutputStride *
                            mep_cctv::core::kNumAnchors),
      0.0f);
}

// Helper: inject a single detection into a tensor.
static void InjectDetection(std::vector<float>& tensor, int anchor, float cx,
                            float cy, float bw, float bh, float score,
                            int cls) {
  const int n = mep_cctv::core::kNumAnchors;
  tensor[static_cast<unsigned>(0 * n + anchor)] = cx;
  tensor[static_cast<unsigned>(1 * n + anchor)] = cy;
  tensor[static_cast<unsigned>(2 * n + anchor)] = bw;
  tensor[static_cast<unsigned>(3 * n + anchor)] = bh;
  tensor[static_cast<unsigned>((4 + cls) * n + anchor)] = score;
}

// Identity transform for direct decoder tests, independent of resize geometry.
static mep_cctv::core::LetterboxParams IdentityParams() {
  return {1.0f, 0.0f, 0.0f};
}

// ---------------------------------------------------------------------------
// DecodeYoloOutput tests
// ---------------------------------------------------------------------------

TEST(DecodeYoloOutputTest, ZeroTensorNoDetections) {
  const auto tensor = ZeroTensor();
  const auto boxes = mep_cctv::core::DecodeYoloOutput(
      tensor.data(), IdentityParams(), 640, 640, 0.25f, 0.45f);
  EXPECT_TRUE(boxes.empty());
}

TEST(DecodeYoloOutputTest, SingleHighScoreDetection) {
  auto tensor = ZeroTensor();
  // Anchor 10: cx=320, cy=320, w=100, h=100, score=0.9, class=0
  InjectDetection(tensor, 10, 320.0f, 320.0f, 100.0f, 100.0f, 0.9f, 0);

  const auto boxes = mep_cctv::core::DecodeYoloOutput(
      tensor.data(), IdentityParams(), 640, 640, 0.25f, 0.45f);

  ASSERT_EQ(1, static_cast<int>(boxes.size()));
  EXPECT_NEAR(0.9f, boxes[0].score, 1e-5f);
  EXPECT_EQ(0, boxes[0].class_id);
  // cx=320, w=100 -> x=320-50=270
  EXPECT_NEAR(270.0f, boxes[0].x, 0.5f);
  EXPECT_NEAR(270.0f, boxes[0].y, 0.5f);
  EXPECT_NEAR(100.0f, boxes[0].w, 0.5f);
  EXPECT_NEAR(100.0f, boxes[0].h, 0.5f);
}

TEST(DecodeYoloOutputTest, ScoreBelowThresholdFiltered) {
  auto tensor = ZeroTensor();
  InjectDetection(tensor, 10, 320.0f, 320.0f, 100.0f, 100.0f, 0.1f, 0);

  const auto boxes = mep_cctv::core::DecodeYoloOutput(
      tensor.data(), IdentityParams(), 640, 640, 0.25f, 0.45f);
  EXPECT_TRUE(boxes.empty());
}

TEST(DecodeYoloOutputTest, TwoIdenticalBoxesNmsSuppressesOne) {
  auto tensor = ZeroTensor();
  InjectDetection(tensor, 10, 320.0f, 320.0f, 100.0f, 100.0f, 0.9f, 0);
  InjectDetection(tensor, 11, 320.0f, 320.0f, 100.0f, 100.0f, 0.8f, 0);

  const auto boxes = mep_cctv::core::DecodeYoloOutput(
      tensor.data(), IdentityParams(), 640, 640, 0.25f, 0.45f);
  ASSERT_EQ(1, static_cast<int>(boxes.size()));
  EXPECT_NEAR(0.9f, boxes[0].score, 1e-5f);
}

TEST(DecodeYoloOutputTest, TwoDisjointBoxesBothKept) {
  auto tensor = ZeroTensor();
  InjectDetection(tensor, 10, 100.0f, 100.0f, 50.0f, 50.0f, 0.9f, 0);
  InjectDetection(tensor, 20, 500.0f, 500.0f, 50.0f, 50.0f, 0.8f, 1);

  const auto boxes = mep_cctv::core::DecodeYoloOutput(
      tensor.data(), IdentityParams(), 640, 640, 0.25f, 0.45f);
  ASSERT_EQ(2, static_cast<int>(boxes.size()));
}

TEST(DecodeYoloOutputTest, CoordinatesUnletterboxed) {
  // Letterbox a 1280x720 frame, then inject a box at the centre of the
  // letterbox and check it maps back to the centre of the original frame.
  // Expressed in terms of kInputSize so the test holds at any resolution.
  const auto params = mep_cctv::core::ComputeLetterboxParams(1280, 720);
  const float kSize = static_cast<float>(mep_cctv::core::kInputSize);
  const float kCentre = kSize * 0.5f;
  const float kBox = kSize * 0.15625f;  // 100 px at 640

  auto tensor = ZeroTensor();
  InjectDetection(tensor, 5, kCentre, kCentre, kBox, kBox, 0.9f, 0);

  const auto boxes = mep_cctv::core::DecodeYoloOutput(tensor.data(), params,
                                                      1280, 720, 0.25f, 0.45f);

  ASSERT_EQ(1, static_cast<int>(boxes.size()));
  // The box is unscaled back into original-frame pixels: kBox / scale.
  const float expected = kBox / params.scale;
  EXPECT_NEAR(expected, boxes[0].w, 1.0f);
  EXPECT_NEAR(expected, boxes[0].h, 1.0f);
}

TEST(DecodeYoloOutputTest, NegativeCoordsClampedToZero) {
  // Inject a box near the origin that would produce negative top-left.
  auto tensor = ZeroTensor();
  // cx=5, cy=5, w=50, h=50 -> x = 5-25 = -20 -> clamped to 0
  InjectDetection(tensor, 5, 5.0f, 5.0f, 50.0f, 50.0f, 0.9f, 0);

  const auto boxes = mep_cctv::core::DecodeYoloOutput(
      tensor.data(), IdentityParams(), 640, 640, 0.25f, 0.45f);

  ASSERT_EQ(1, static_cast<int>(boxes.size()));
  EXPECT_GE(boxes[0].x, 0.0f);
  EXPECT_GE(boxes[0].y, 0.0f);
}

TEST(DecodeYoloOutputTest, BoxCoordsClampedToImageBoundary) {
  // Box extending past image edge.
  auto tensor = ZeroTensor();
  InjectDetection(tensor, 5, 630.0f, 630.0f, 100.0f, 100.0f, 0.9f, 0);

  const auto boxes = mep_cctv::core::DecodeYoloOutput(
      tensor.data(), IdentityParams(), 640, 640, 0.25f, 0.45f);

  ASSERT_EQ(1, static_cast<int>(boxes.size()));
  EXPECT_LE(boxes[0].x + boxes[0].w, 640.0f);
  EXPECT_LE(boxes[0].y + boxes[0].h, 640.0f);
}

TEST(DecodeYoloOutputTest, HighThresholdFiltersAll) {
  auto tensor = ZeroTensor();
  InjectDetection(tensor, 10, 320.0f, 320.0f, 100.0f, 100.0f, 0.9f, 0);

  const auto boxes = mep_cctv::core::DecodeYoloOutput(
      tensor.data(), IdentityParams(), 640, 640, 0.99f, 0.45f);
  EXPECT_TRUE(boxes.empty());
}

}  // namespace
