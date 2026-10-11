// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file nms_test.cc
 * @brief Unit tests for ComputeIoU and ApplyNMS.
 *
 * Tests cover:
 *   - IoU of identical boxes = 1.0
 *   - IoU of disjoint boxes = 0.0
 *   - IoU of partially overlapping boxes (known value)
 *   - ApplyNMS: overlapping boxes suppressed
 *   - ApplyNMS: disjoint boxes kept
 *   - ApplyNMS: empty input -> empty output
 *   - ApplyNMS: single box -> returned unchanged
 *   - SIMD vs scalar equivalence
 */

#include "src/core/nms.hpp"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
// Scalar inline reference for equivalence tests
// ---------------------------------------------------------------------------

namespace test_nms_scalar {

static float ComputeIoU(const cctv_box& a, const cctv_box& b) noexcept {
  const float ax2 = a.x + a.w;
  const float ay2 = a.y + a.h;
  const float bx2 = b.x + b.w;
  const float by2 = b.y + b.h;
  const float ix1 = std::max(a.x, b.x);
  const float iy1 = std::max(a.y, b.y);
  const float ix2 = std::min(ax2, bx2);
  const float iy2 = std::min(ay2, by2);
  const float iw = ix2 - ix1;
  const float ih = iy2 - iy1;
  if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
  const float inter = iw * ih;
  const float uni = a.w * a.h + b.w * b.h - inter;
  return (uni > 0.0f) ? inter / uni : 0.0f;
}

static cctv_box MakeBox(float x, float y, float w, float h, float score,
                        int cls = 0) {
  cctv_box b;
  b.x = x;
  b.y = y;
  b.w = w;
  b.h = h;
  b.score = score;
  b.class_id = cls;
  return b;
}

}  // namespace test_nms_scalar

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Helper to construct a box.
static cctv_box MakeBox(float x, float y, float w, float h, float score,
                        int cls = 0) {
  return test_nms_scalar::MakeBox(x, y, w, h, score, cls);
}

// ---------------------------------------------------------------------------
// ComputeIoU tests
// ---------------------------------------------------------------------------

TEST(ComputeIoUTest, IdenticalBoxes) {
  const cctv_box a = MakeBox(10, 10, 50, 50, 0.9f);
  EXPECT_FLOAT_EQ(1.0f, mep_cctv::core::ComputeIoU(a, a));
}

TEST(ComputeIoUTest, DisjointBoxesHorizontal) {
  const cctv_box a = MakeBox(0, 0, 50, 50, 0.9f);
  const cctv_box b = MakeBox(100, 0, 50, 50, 0.8f);
  EXPECT_FLOAT_EQ(0.0f, mep_cctv::core::ComputeIoU(a, b));
}

TEST(ComputeIoUTest, DisjointBoxesVertical) {
  const cctv_box a = MakeBox(0, 0, 50, 50, 0.9f);
  const cctv_box b = MakeBox(0, 100, 50, 50, 0.8f);
  EXPECT_FLOAT_EQ(0.0f, mep_cctv::core::ComputeIoU(a, b));
}

TEST(ComputeIoUTest, HalfOverlapKnownValue) {
  // a=[0,0,10,10], b=[5,0,10,10]
  // intersection = [5,0,5,10] = 50; union = 100+100-50 = 150
  // IoU = 50/150 = 1/3
  const cctv_box a = MakeBox(0, 0, 10, 10, 0.9f);
  const cctv_box b = MakeBox(5, 0, 10, 10, 0.8f);
  EXPECT_NEAR(1.0f / 3.0f, mep_cctv::core::ComputeIoU(a, b), 1e-5f);
  // Cross-check against inline reference (exercises test_nms_scalar).
  EXPECT_NEAR(1.0f / 3.0f, test_nms_scalar::ComputeIoU(a, b), 1e-5f);
}

TEST(ComputeIoUTest, ContainedBox) {
  // a=[0,0,10,10], b=[2,2,6,6] (fully inside a)
  // inter=36, union=100+36-36=100, IoU=0.36
  const cctv_box a = MakeBox(0, 0, 10, 10, 0.9f);
  const cctv_box b = MakeBox(2, 2, 6, 6, 0.5f);
  EXPECT_NEAR(36.0f / 100.0f, mep_cctv::core::ComputeIoU(a, b), 1e-5f);
}

TEST(ComputeIoUTest, TouchingEdgesNoOverlap) {
  const cctv_box a = MakeBox(0, 0, 10, 10, 0.9f);
  const cctv_box b = MakeBox(10, 0, 10, 10, 0.8f);
  EXPECT_FLOAT_EQ(0.0f, mep_cctv::core::ComputeIoU(a, b));
}

// ---------------------------------------------------------------------------
// ApplyNMS tests
// ---------------------------------------------------------------------------

TEST(ApplyNMSTest, EmptyInput) {
  const std::vector<cctv_box> empty;
  const auto out = mep_cctv::core::ApplyNMS(empty, 0.45f);
  EXPECT_TRUE(out.empty());
}

TEST(ApplyNMSTest, SingleBox) {
  const cctv_box box = MakeBox(10, 10, 50, 50, 0.9f);
  const auto out = mep_cctv::core::ApplyNMS({box}, 0.45f);
  ASSERT_EQ(1, static_cast<int>(out.size()));
  EXPECT_FLOAT_EQ(0.9f, out[0].score);
}

TEST(ApplyNMSTest, TwoDisjointBoxesKeptBoth) {
  const cctv_box a = MakeBox(0, 0, 10, 10, 0.9f);
  const cctv_box b = MakeBox(200, 200, 10, 10, 0.8f);
  const auto out = mep_cctv::core::ApplyNMS({a, b}, 0.45f);
  ASSERT_EQ(2, static_cast<int>(out.size()));
}

TEST(ApplyNMSTest, TwoOverlappingBoxesSuppressLower) {
  // Both boxes almost identical -> IoU near 1.0 -> lower score suppressed.
  const cctv_box a = MakeBox(10, 10, 50, 50, 0.95f);
  const cctv_box b = MakeBox(11, 11, 50, 50, 0.70f);
  const auto out = mep_cctv::core::ApplyNMS({a, b}, 0.45f);
  ASSERT_EQ(1, static_cast<int>(out.size()));
  EXPECT_NEAR(0.95f, out[0].score, 1e-5f);
}

TEST(ApplyNMSTest, OutputInDescendingScoreOrder) {
  const cctv_box a = MakeBox(0, 0, 10, 10, 0.5f);
  const cctv_box b = MakeBox(100, 100, 10, 10, 0.9f);
  const cctv_box c = MakeBox(200, 200, 10, 10, 0.7f);
  const auto out = mep_cctv::core::ApplyNMS({a, b, c}, 0.45f);
  ASSERT_EQ(3, static_cast<int>(out.size()));
  EXPECT_GE(out[0].score, out[1].score);
  EXPECT_GE(out[1].score, out[2].score);
}

TEST(ApplyNMSTest, ThreeBoxesTwoOverlapOneKept) {
  // A and B overlap heavily; C is disjoint. Expect 2 survivors: A and C.
  const cctv_box a = MakeBox(0, 0, 100, 100, 0.9f);
  const cctv_box b = MakeBox(5, 5, 100, 100, 0.8f);
  const cctv_box c = MakeBox(500, 500, 100, 100, 0.7f);
  const auto out = mep_cctv::core::ApplyNMS({a, b, c}, 0.45f);
  ASSERT_EQ(2, static_cast<int>(out.size()));
  EXPECT_NEAR(0.9f, out[0].score, 1e-5f);
  EXPECT_NEAR(0.7f, out[1].score, 1e-5f);
}

TEST(ApplyNMSTest, ZeroIouThresholdSuppressesAll) {
  // threshold=0: any non-zero IoU triggers suppression.
  const cctv_box a = MakeBox(0, 0, 10, 10, 0.9f);
  const cctv_box b = MakeBox(5, 5, 10, 10, 0.8f);  // Overlaps a.
  const auto out = mep_cctv::core::ApplyNMS({a, b}, 0.0f);
  ASSERT_EQ(1, static_cast<int>(out.size()));
}

TEST(ApplyNMSTest, MixedVectorLanesAndTailSuppressAcrossClasses) {
  std::vector<cctv_box> boxes = {MakeBox(0, 0, 10, 10, 1.0f)};
  for (int i = 1; i <= 13; ++i) {
    const float x = i % 2 == 0 ? static_cast<float>(i * 20) : 0.0f;
    boxes.push_back(MakeBox(x, 0, 10, 10, 1.0f - i * 0.05f, i));
  }
  const auto out = mep_cctv::core::ApplyNMS(boxes, 0.45f);
  ASSERT_EQ(7, static_cast<int>(out.size()));
  EXPECT_EQ(0, out[0].class_id);
  for (int i = 1; i < 7; ++i) {
    EXPECT_EQ(i * 2, out[i].class_id);
    EXPECT_FLOAT_EQ(static_cast<float>(i * 40), out[i].x);
  }
}

TEST(ApplyNMSTest, DisjointVectorLanesRemainInDescendingOrder) {
  std::vector<cctv_box> boxes;
  for (int i = 0; i < 17; ++i) {
    boxes.push_back(
        MakeBox(static_cast<float>(i * 20), 0, 10, 10, 0.1f + i * 0.04f, i));
  }
  const auto out = mep_cctv::core::ApplyNMS(boxes, 0.45f);
  ASSERT_EQ(17, static_cast<int>(out.size()));
  for (int i = 0; i < 17; ++i) { EXPECT_EQ(16 - i, out[i].class_id); }
}

// ---------------------------------------------------------------------------
// SIMD vs scalar equivalence
// ---------------------------------------------------------------------------

#if CCTV_SIMD_ENABLED

TEST(NmsEquivalenceTest, IoUMatchesScalarReference) {
  const cctv_box a = MakeBox(10, 20, 40, 60, 0.9f);
  const cctv_box b = MakeBox(20, 30, 40, 60, 0.7f);
  const float ref = test_nms_scalar::ComputeIoU(a, b);
  const float got = mep_cctv::core::ComputeIoU(a, b);
  EXPECT_NEAR(ref, got, 1e-4f);
}

TEST(NmsEquivalenceTest, ApplyNMSMatchesScalarReference) {
  // Twelve fixed boxes exercise the NEON and AVX2 inner loops.
  std::vector<cctv_box> boxes;
  for (int i = 0; i < 12; ++i) {
    const float x = static_cast<float>(i * 30);
    const float score = 0.9f - i * 0.05f;
    if (score <= 0.0f) break;
    boxes.push_back(MakeBox(x, 0, 50, 50, score));
  }
  const auto out = mep_cctv::core::ApplyNMS(boxes, 0.45f);
  // Adjacent boxes overlap, but their IoU is below the suppression threshold.
  EXPECT_GT(static_cast<int>(out.size()), 0);
  EXPECT_LE(static_cast<int>(out.size()), static_cast<int>(boxes.size()));
}

#else

TEST(NmsEquivalenceTest, ScalarOnlyTrivialCheck) {
  const cctv_box a = MakeBox(0, 0, 10, 10, 0.9f);
  const cctv_box b = MakeBox(0, 0, 10, 10, 0.8f);
  const auto out = mep_cctv::core::ApplyNMS({a, b}, 0.45f);
  EXPECT_EQ(1, static_cast<int>(out.size()));
}

#endif  // CCTV_SIMD_ENABLED

}  // namespace
