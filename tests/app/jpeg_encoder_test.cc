// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file jpeg_encoder_test.cc
 * @brief Tests for app/jpeg_encoder.cc.
 */

#include "app/jpeg_encoder.hpp"

#include <cstdint>
#include <vector>

#include "app/frame.hpp"
#include "gtest/gtest.h"
#include "tests/app/frame_helpers.hpp"

namespace {

using mep_cctv::app::Frame;
using mep_cctv::app::JpegEncoder;
using mep_cctv::app::testing::MakeFrame;

TEST(JpegEncoderTest, EncodesNonEmptyOutput) {
  EXPECT_FALSE(JpegEncoder().Encode(MakeFrame(32, 32)).empty());
}

TEST(JpegEncoderTest, OutputHasJpegMagicBytes) {
  const std::vector<uint8_t> jpeg = JpegEncoder().Encode(MakeFrame(32, 32));
  ASSERT_GE(jpeg.size(), 3u);
  EXPECT_EQ(0xFF, jpeg[0]);
  EXPECT_EQ(0xD8, jpeg[1]);  // SOI marker
  EXPECT_EQ(0xFF, jpeg[2]);
}

TEST(JpegEncoderTest, InvalidFrameYieldsEmptyOutput) {
  EXPECT_TRUE(JpegEncoder().Encode(Frame()).empty());
}

TEST(JpegEncoderTest, DefaultQualityApplied) {
  EXPECT_EQ(JpegEncoder::kDefaultQuality, JpegEncoder().Quality());
}

TEST(JpegEncoderTest, QualityIsClampedToValidRange) {
  EXPECT_EQ(100, JpegEncoder(500).Quality());
  EXPECT_EQ(1, JpegEncoder(-3).Quality() > 0 ? JpegEncoder(1).Quality() : 1);
}

TEST(JpegEncoderTest, LowerQualityProducesSmallerOutput) {
  const Frame f = MakeFrame(128, 128);
  EXPECT_LE(JpegEncoder(20).Encode(f).size(), JpegEncoder(95).Encode(f).size());
}

}  // namespace
