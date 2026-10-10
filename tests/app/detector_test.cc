// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file detector_test.cc
 * @brief Tests for app/detector.cc -- the app's view of the public C ABI.
 */

#include "app/detector.hpp"

#include <string>
#include <vector>

#include "app/frame.hpp"
#include "app/sources.hpp"
#include "gtest/gtest.h"
#include "tests/app/frame_helpers.hpp"

namespace {

using mep_cctv::app::Detector;
using mep_cctv::app::Frame;
using mep_cctv::app::SyntheticVideoSource;
using mep_cctv::app::testing::MakeFrame;

TEST(DetectorTest, ReportsBackendName) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  EXPECT_EQ(std::string(CCTV_BACKEND_NAME), det.BackendName());
}

TEST(DetectorTest, ReportsVersion) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  EXPECT_EQ(std::string(CCTV_VERSION), det.Version());
}

TEST(DetectorTest, ReportsCompiledSimdSetting) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  EXPECT_EQ(CCTV_SIMD_ENABLED != 0, det.SimdEnabled());
}

TEST(DetectorTest, InvalidFrameYieldsNoDetections) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  EXPECT_TRUE(det.DetectObjects(Frame()).empty());
}

TEST(DetectorTest, EmptyAudioIsRejected) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  const cctv_audio_result r = det.DetectAudio(std::vector<float>(), 16000);
  EXPECT_EQ(0, r.person_present);
}

TEST(DetectorTest, ZeroSampleRateIsRejected) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  const cctv_audio_result r = det.DetectAudio(std::vector<float>(16, 0.5f), 0);
  EXPECT_EQ(0, r.person_present);
}

// Native engines need no external ONNX model path.
#if defined(CCTV_USE_NATIVE_BACKEND) && CCTV_USE_NATIVE_BACKEND
TEST(DetectorTest, NativeBackendNeedsNoModelFiles) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  EXPECT_TRUE(det.IsReady());
}

TEST(DetectorTest, AudioDistinguishesSilenceFromAlternatingSamples) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  ASSERT_TRUE(det.IsReady());
  EXPECT_EQ(
      0, det.DetectAudio(std::vector<float>(16, 0.0f), 16000).person_present);
  std::vector<float> samples(16, 0.5f);
  for (int i = 0; i < 16; i += 2) { samples[i] = -0.5f; }
  const auto result = det.DetectAudio(samples, 16000);
  EXPECT_EQ(1, result.person_present);
  EXPECT_NEAR(0.5f, result.score, 1e-5f);
}

TEST(DetectorTest, DetectsOnSyntheticFrameWithoutCrashing) {
  const Detector det("", Detector::kDefaultConfThreshold,
                     Detector::kDefaultIouThreshold);
  ASSERT_TRUE(det.IsReady());
  SyntheticVideoSource src(320, 240);
  const std::vector<cctv_box> boxes = det.DetectObjects(src.Read());
  for (const cctv_box& b : boxes) {
    EXPECT_GE(b.score, 0.0f);
    EXPECT_LE(b.score, 1.0f);
  }
}
#endif  // CCTV_USE_NATIVE_BACKEND

}  // namespace
