// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file sources_test.cc
 * @brief Tests for app/sources.cc -- the synthetic stand-ins for the
 *        camera and microphone, which keep the suite hardware-free.
 */

#include "app/sources.hpp"

#include "gtest/gtest.h"

namespace {

using mep_cctv::app::Frame;
using mep_cctv::app::SilentAudioSource;
using mep_cctv::app::SyntheticVideoSource;

TEST(SyntheticVideoSourceTest, ProducesValidFrame) {
  SyntheticVideoSource src(64, 48);
  EXPECT_TRUE(src.Read().IsValid());
}

TEST(SyntheticVideoSourceTest, HonoursRequestedGeometry) {
  SyntheticVideoSource src(80, 60);
  const Frame f = src.Read();
  EXPECT_EQ(80, f.width);
  EXPECT_EQ(60, f.height);
}

TEST(SyntheticVideoSourceTest, SuccessiveFramesDiffer) {
  // The page must visibly update, so the pattern has to animate.
  SyntheticVideoSource src(64, 48);
  const Frame a = src.Read();
  const Frame b = src.Read();
  EXPECT_NE(a.rgb, b.rgb);
}

TEST(SyntheticVideoSourceTest, ReportsItsName) {
  EXPECT_EQ("synthetic", SyntheticVideoSource(8, 8).Name());
}

// ---------------------------------------------------------------------------
// SilentAudioSource
// ---------------------------------------------------------------------------

TEST(SilentAudioSourceTest, ReturnsRequestedChunkSize) {
  SilentAudioSource src(16000, 512);
  EXPECT_EQ(512u, src.Read().size());
}

TEST(SilentAudioSourceTest, ReturnsSilence) {
  SilentAudioSource src(16000, 32);
  for (const float s : src.Read()) { EXPECT_FLOAT_EQ(0.0f, s); }
}

TEST(SilentAudioSourceTest, ReportsSampleRate) {
  EXPECT_EQ(16000, SilentAudioSource(16000, 32).SampleRate());
}

}  // namespace
