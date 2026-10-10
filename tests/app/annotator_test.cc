// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file annotator_test.cc
 * @brief Tests for app/annotator.cc -- the "mark the object with a square"
 *        requirement, plus per-object colour and labels.
 */

#include "app/annotator.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "app/frame.hpp"
#include "gtest/gtest.h"
#include "tests/app/frame_helpers.hpp"

namespace {

using mep_cctv::app::Annotator;
using mep_cctv::app::Frame;
using mep_cctv::app::testing::MakeBox;
using mep_cctv::app::testing::MakeFrame;

TEST(AnnotatorTest, DrawsBoxIntoFrame) {
  Frame f = MakeFrame(64, 64);
  const cctv_box box = MakeBox(10.0f, 10.0f, 20.0f, 20.0f);
  EXPECT_EQ(1, Annotator().Draw(&f, &box, 1));
}

TEST(AnnotatorTest, DrawnPixelsChangeTheFrame) {
  Frame f = MakeFrame(64, 64);
  const std::vector<uint8_t> before = f.rgb;
  const cctv_box box = MakeBox(10.0f, 10.0f, 20.0f, 20.0f);
  Annotator().Draw(&f, &box, 1);
  EXPECT_NE(before, f.rgb);
}

TEST(AnnotatorTest, BoxEdgeIsGreen) {
  Frame f = MakeFrame(64, 64);
  const cctv_box box = MakeBox(10.0f, 10.0f, 20.0f, 20.0f);
  Annotator().Draw(&f, &box, 1);
  const std::size_t i = (10u * 64u + 15u) * 3u;  // on the top edge
  EXPECT_EQ(0u, f.rgb[i]);
  EXPECT_EQ(255u, f.rgb[i + 1]);
  EXPECT_EQ(0u, f.rgb[i + 2]);
}

TEST(AnnotatorTest, PixelFarFromBoxIsUntouched) {
  Frame f = MakeFrame(64, 64);
  const cctv_box box = MakeBox(10.0f, 10.0f, 8.0f, 8.0f);
  Annotator().Draw(&f, &box, 1);
  const std::size_t i = (60u * 64u + 60u) * 3u;
  EXPECT_EQ(128u, f.rgb[i]);
}

TEST(AnnotatorTest, ZeroBoxesDrawsNothing) {
  Frame f = MakeFrame(32, 32);
  const std::vector<uint8_t> before = f.rgb;
  Annotator().Draw(&f, nullptr, 0);
  EXPECT_EQ(before, f.rgb);
}

TEST(AnnotatorTest, BoxFullyOutsideFrameIsSkipped) {
  Frame f = MakeFrame(32, 32);
  const cctv_box box = MakeBox(500.0f, 500.0f, 20.0f, 20.0f);
  EXPECT_EQ(0, Annotator().Draw(&f, &box, 1));
}

TEST(AnnotatorTest, BoxStraddlingEdgeIsClippedNotCrashing) {
  // Regression guard: a detection near the border must not write out of
  // bounds. Under ASan this would trap rather than merely fail.
  Frame f = MakeFrame(32, 32);
  const cctv_box box = MakeBox(-5.0f, -5.0f, 40.0f, 40.0f);
  EXPECT_EQ(1, Annotator().Draw(&f, &box, 1));
  EXPECT_TRUE(f.IsValid());
}

TEST(AnnotatorTest, NegativeCoordinatesHandled) {
  Frame f = MakeFrame(32, 32);
  const cctv_box box = MakeBox(-100.0f, -100.0f, 50.0f, 50.0f);
  Annotator().Draw(&f, &box, 1);
  EXPECT_TRUE(f.IsValid());
}

TEST(AnnotatorTest, InvalidFrameIsRejected) {
  Frame f;
  const cctv_box box = MakeBox(1.0f, 1.0f, 2.0f, 2.0f);
  EXPECT_EQ(0, Annotator().Draw(&f, &box, 1));
}

TEST(AnnotatorTest, NullFrameIsRejected) {
  const cctv_box box = MakeBox(1.0f, 1.0f, 2.0f, 2.0f);
  EXPECT_EQ(0, Annotator().Draw(nullptr, &box, 1));
}

TEST(AnnotatorTest, DrawsMultipleBoxes) {
  Frame f = MakeFrame(64, 64);
  const cctv_box boxes[2] = {MakeBox(2.0f, 2.0f, 10.0f, 10.0f),
                             MakeBox(30.0f, 30.0f, 10.0f, 10.0f)};
  EXPECT_EQ(2, Annotator().Draw(&f, boxes, 2));
}

/** @brief Read one pixel as {r,g,b}. */
std::vector<uint8_t> PixelAt(const Frame& f, int x, int y) {
  const std::size_t i =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(f.width) +
       static_cast<std::size_t>(x)) *
      3u;
  return {f.rgb[i], f.rgb[i + 1], f.rgb[i + 2]};
}

// The reported problem: with every box the same colour, two adjacent
// detections read as one shape. Consecutive boxes must differ.
TEST(AnnotatorColorTest, SecondBoxUsesADifferentColour) {
  Frame f = MakeFrame(200, 120);
  const cctv_box boxes[2] = {MakeBox(10.0f, 40.0f, 40.0f, 40.0f),
                             MakeBox(100.0f, 40.0f, 40.0f, 40.0f)};
  ASSERT_EQ(2, Annotator().Draw(&f, boxes, 2));
  const std::vector<uint8_t> first = PixelAt(f, 30, 40);
  const std::vector<uint8_t> second = PixelAt(f, 120, 40);
  EXPECT_NE(first, second);
}

// The first box stays green so the familiar single-object view is unchanged.
TEST(AnnotatorColorTest, FirstBoxIsStillGreen) {
  Frame f = MakeFrame(200, 120);
  const cctv_box box = MakeBox(10.0f, 40.0f, 40.0f, 40.0f);
  Annotator().Draw(&f, &box, 1);
  const std::vector<uint8_t> px = PixelAt(f, 30, 40);
  EXPECT_EQ(std::vector<uint8_t>({0u, 255u, 0u}), px);
}

// More detections than palette entries must keep working, not read past it.
TEST(AnnotatorColorTest, ManyBoxesWrapAroundThePalette) {
  Frame f = MakeFrame(400, 300);
  std::vector<cctv_box> boxes;
  for (int i = 0; i < 40; ++i) {
    boxes.push_back(MakeBox(static_cast<float>(i % 8) * 40.0f + 2.0f,
                            static_cast<float>(i / 8) * 50.0f + 2.0f, 30.0f,
                            30.0f));
  }
  EXPECT_EQ(40,
            Annotator().Draw(&f, boxes.data(), static_cast<int>(boxes.size())));
  EXPECT_TRUE(f.IsValid());
}

TEST(AnnotatorLabelTest, LabelPixelsAppearAboveTheBox) {
  Frame f = MakeFrame(200, 120);
  const std::vector<uint8_t> before = f.rgb;
  // Leave room above the box so the label band is inside the frame.
  const cctv_box box = MakeBox(20.0f, 60.0f, 60.0f, 40.0f);
  Annotator().Draw(&f, &box, 1);

  bool changed_above = false;
  for (int y = 40; y < 60 && !changed_above; ++y) {
    for (int x = 20; x < 80; ++x) {
      if (PixelAt(f, x, y) != std::vector<uint8_t>({128u, 128u, 128u})) {
        changed_above = true;
        break;
      }
    }
  }
  EXPECT_TRUE(changed_above) << "no label was drawn above the rectangle";
  EXPECT_NE(before, f.rgb);
}

// A detection at the very top has no room above it; the label must move
// inside the box rather than being clipped away or writing out of bounds.
TEST(AnnotatorLabelTest, LabelAtTopEdgeStaysInsideTheFrame) {
  Frame f = MakeFrame(200, 120);
  const cctv_box box = MakeBox(10.0f, 0.0f, 80.0f, 50.0f);
  EXPECT_EQ(1, Annotator().Draw(&f, &box, 1));
  EXPECT_TRUE(f.IsValid());

  bool changed_inside = false;
  for (int y = 0; y < 20 && !changed_inside; ++y) {
    for (int x = 10; x < 90; ++x) {
      if (PixelAt(f, x, y) != std::vector<uint8_t>({128u, 128u, 128u})) {
        changed_inside = true;
        break;
      }
    }
  }
  EXPECT_TRUE(changed_inside);
}

// An unknown class must still render something rather than crashing.
TEST(AnnotatorLabelTest, UnknownClassStillDraws) {
  Frame f = MakeFrame(200, 120);
  cctv_box box = MakeBox(20.0f, 60.0f, 60.0f, 40.0f);
  box.class_id = 12345;
  EXPECT_EQ(1, Annotator().Draw(&f, &box, 1));
  EXPECT_TRUE(f.IsValid());
}

}  // namespace
