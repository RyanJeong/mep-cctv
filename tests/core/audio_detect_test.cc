// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file audio_detect_test.cc
 * @brief Tests for the code-based acoustic activity detector.
 *
 * These run identically in both modes: audio never touches ONNX or
 * TVM, which is what keeps the mode benchmark measuring only the image model.
 */

#include "src/core/audio_detect.hpp"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace {

using mep_cctv::core::DetectAcousticActivity;
using mep_cctv::core::kAudioRmsThreshold;

/** @brief Build a sine tone of the given amplitude. */
std::vector<float> Tone(int n, float amplitude) {
  std::vector<float> samples(static_cast<unsigned>(n), 0.0f);
  for (int i = 0; i < n; ++i) {
    samples[static_cast<unsigned>(i)] =
        amplitude * std::sin(2.0f * 3.14159265f * 440.0f *
                             static_cast<float>(i) / 16000.0f);
  }
  return samples;
}

// ---------------------------------------------------------------------------
// Contract / edge cases
// ---------------------------------------------------------------------------

TEST(AudioDetectTest, NullSamplesReportNothing) {
  const cctv_audio_result r = DetectAcousticActivity(nullptr, 0);
  EXPECT_EQ(0, r.person_present);
  EXPECT_FLOAT_EQ(0.0f, r.score);
}

TEST(AudioDetectTest, ZeroLengthReportsNothing) {
  const std::vector<float> s(16, 0.5f);
  EXPECT_EQ(0, DetectAcousticActivity(s.data(), 0).person_present);
}

TEST(AudioDetectTest, NegativeLengthReportsNothing) {
  const std::vector<float> s(16, 0.5f);
  EXPECT_EQ(0, DetectAcousticActivity(s.data(), -5).person_present);
}

TEST(AudioDetectTest, SingleSampleIsHandled) {
  const float one = 0.9f;
  // A single sample has zero variance once the mean is removed.
  EXPECT_EQ(0, DetectAcousticActivity(&one, 1).person_present);
}

// ---------------------------------------------------------------------------
// Behaviour
// ---------------------------------------------------------------------------

TEST(AudioDetectTest, SilenceIsNotActivity) {
  const std::vector<float> silence(16000, 0.0f);
  const cctv_audio_result r =
      DetectAcousticActivity(silence.data(), static_cast<int>(silence.size()));
  EXPECT_EQ(0, r.person_present);
  EXPECT_NEAR(0.0f, r.score, 1e-6f);
}

TEST(AudioDetectTest, LoudToneIsActivity) {
  const std::vector<float> loud = Tone(16000, 0.5f);
  const cctv_audio_result r =
      DetectAcousticActivity(loud.data(), static_cast<int>(loud.size()));
  EXPECT_NE(0, r.person_present);
  EXPECT_GT(r.score, kAudioRmsThreshold);
}

TEST(AudioDetectTest, VeryQuietToneIsNotActivity) {
  const std::vector<float> quiet = Tone(16000, 0.001f);
  EXPECT_EQ(0, DetectAcousticActivity(quiet.data(), 16000).person_present);
}

TEST(AudioDetectTest, LouderInputScoresHigher) {
  const std::vector<float> soft = Tone(16000, 0.10f);
  const std::vector<float> loud = Tone(16000, 0.40f);
  EXPECT_LT(DetectAcousticActivity(soft.data(), 16000).score,
            DetectAcousticActivity(loud.data(), 16000).score);
}

TEST(AudioDetectTest, ConstantDcOffsetIsNotActivity) {
  // A constant microphone bias is not acoustic activity.
  const std::vector<float> dc(16000, 0.8f);
  const cctv_audio_result r = DetectAcousticActivity(dc.data(), 16000);
  EXPECT_EQ(0, r.person_present);
  EXPECT_NEAR(0.0f, r.score, 1e-5f);
}

TEST(AudioDetectTest, ToneOnTopOfDcOffsetIsStillActivity) {
  std::vector<float> mixed = Tone(16000, 0.4f);
  for (float& v : mixed) {
    v += 0.5f;  // add the bias back in
  }
  EXPECT_NE(0, DetectAcousticActivity(mixed.data(), 16000).person_present);
}

TEST(AudioDetectTest, ScoreIsClampedToOne) {
  const std::vector<float> huge = Tone(16000, 50.0f);
  EXPECT_LE(DetectAcousticActivity(huge.data(), 16000).score, 1.0f);
}

TEST(AudioDetectTest, ScoreIsNeverNegative) {
  const std::vector<float> t = Tone(16000, 0.3f);
  EXPECT_GE(DetectAcousticActivity(t.data(), 16000).score, 0.0f);
}

TEST(AudioDetectTest, IsDeterministic) {
  const std::vector<float> t = Tone(16000, 0.3f);
  const cctv_audio_result a = DetectAcousticActivity(t.data(), 16000);
  const cctv_audio_result b = DetectAcousticActivity(t.data(), 16000);
  EXPECT_FLOAT_EQ(a.score, b.score);
  EXPECT_EQ(a.person_present, b.person_present);
}

}  // namespace
