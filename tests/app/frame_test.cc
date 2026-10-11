// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file frame_test.cc
 * @brief Tests for app/frame.hpp.
 */

#include "app/frame.hpp"

#include "gtest/gtest.h"
#include "tests/app/frame_helpers.hpp"

namespace {

using mep_cctv::app::Frame;
using mep_cctv::app::testing::MakeFrame;

TEST(FrameTest, DefaultFrameIsInvalid) { EXPECT_FALSE(Frame().IsValid()); }

TEST(FrameTest, CorrectlySizedFrameIsValid) {
  EXPECT_TRUE(MakeFrame(8, 4).IsValid());
}

TEST(FrameTest, MismatchedBufferIsInvalid) {
  Frame f = MakeFrame(8, 4);
  f.rgb.pop_back();
  EXPECT_FALSE(f.IsValid());
}

TEST(FrameTest, ZeroWidthIsInvalid) {
  Frame f = MakeFrame(8, 4);
  f.width = 0;
  EXPECT_FALSE(f.IsValid());
}

}  // namespace
