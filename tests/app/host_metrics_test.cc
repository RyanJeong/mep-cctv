// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file host_metrics_test.cc
 * @brief Tests for app/host_metrics.cc.
 *
 * The rate arithmetic is tested through the pure CpuPercentBetween so the
 * cases are deterministic and need no elapsed real time.
 */

#include "app/host_metrics.hpp"

#include "gtest/gtest.h"

namespace {

using mep_cctv::app::CpuPercentBetween;
using mep_cctv::app::HostMetrics;
using mep_cctv::app::HostSample;

using mep_cctv::app::CpuPercentBetween;
using mep_cctv::app::HostMetrics;
using mep_cctv::app::HostSample;

/** @brief A sample with plausible counters, for the pure arithmetic tests. */
HostSample MakeSample(double cpu_seconds, double wall_seconds) {
  HostSample s;
  s.cpu_seconds = cpu_seconds;
  s.wall_seconds = wall_seconds;
  return s;
}

// One core fully busy for one second on a four-core box is a quarter of the
// machine, which is the whole point of normalising by core count: the number
// means the same thing on a Pi and on a laptop.
TEST(CpuPercentTest, OneCoreOfFourIsTwentyFivePercent) {
  const HostSample a = MakeSample(10.0, 100.0);
  const HostSample b = MakeSample(11.0, 101.0);
  EXPECT_NEAR(25.0, CpuPercentBetween(a, b, 4), 0.01);
}

TEST(CpuPercentTest, AllCoresBusyIsOneHundred) {
  const HostSample a = MakeSample(10.0, 100.0);
  const HostSample b = MakeSample(14.0, 101.0);
  EXPECT_NEAR(100.0, CpuPercentBetween(a, b, 4), 0.01);
}

TEST(CpuPercentTest, IdleProcessIsZero) {
  const HostSample a = MakeSample(10.0, 100.0);
  const HostSample b = MakeSample(10.0, 101.0);
  EXPECT_NEAR(0.0, CpuPercentBetween(a, b, 4), 0.01);
}

// Scheduling jitter can make the measured ratio exceed the theoretical
// maximum; reporting 118% would just look broken.
TEST(CpuPercentTest, OvershootIsClampedToOneHundred) {
  const HostSample a = MakeSample(10.0, 100.0);
  const HostSample b = MakeSample(20.0, 101.0);
  EXPECT_NEAR(100.0, CpuPercentBetween(a, b, 4), 0.01);
}

TEST(CpuPercentTest, NoElapsedTimeIsUnavailable) {
  const HostSample a = MakeSample(10.0, 100.0);
  const HostSample b = MakeSample(10.0, 100.0);
  EXPECT_DOUBLE_EQ(HostMetrics::kUnavailable, CpuPercentBetween(a, b, 4));
}

// The first sample of a run has nothing to diff against.
TEST(CpuPercentTest, MissingPreviousIsUnavailable) {
  const HostSample none;
  const HostSample b = MakeSample(10.0, 100.0);
  EXPECT_DOUBLE_EQ(HostMetrics::kUnavailable, CpuPercentBetween(none, b, 4));
}

// A clock that appears to run backwards must not yield a negative percentage.
TEST(CpuPercentTest, BackwardsIntervalIsUnavailable) {
  const HostSample a = MakeSample(10.0, 101.0);
  const HostSample b = MakeSample(11.0, 100.0);
  EXPECT_DOUBLE_EQ(HostMetrics::kUnavailable, CpuPercentBetween(a, b, 4));
}

TEST(CpuPercentTest, ZeroCoresTreatedAsOne) {
  const HostSample a = MakeSample(10.0, 100.0);
  const HostSample b = MakeSample(10.5, 101.0);
  EXPECT_NEAR(50.0, CpuPercentBetween(a, b, 0), 0.01);
}

TEST(CpuPercentTest, EachMissingCounterIsUnavailable) {
  const HostSample valid = MakeSample(10.0, 100.0);
  EXPECT_DOUBLE_EQ(HostMetrics::kUnavailable,
                   CpuPercentBetween(valid, MakeSample(-1.0, 101.0), 4));
  EXPECT_DOUBLE_EQ(HostMetrics::kUnavailable,
                   CpuPercentBetween(MakeSample(10.0, -1.0), valid, 4));
  EXPECT_DOUBLE_EQ(HostMetrics::kUnavailable,
                   CpuPercentBetween(valid, MakeSample(11.0, -1.0), 4));
}

TEST(CpuPercentTest, DecreasingCpuCounterIsUnavailable) {
  EXPECT_DOUBLE_EQ(
      HostMetrics::kUnavailable,
      CpuPercentBetween(MakeSample(11.0, 100.0), MakeSample(10.0, 101.0), 4));
}

TEST(HostMetricsTest, CoreCountIsAtLeastOne) {
  EXPECT_GE(HostMetrics().CoreCount(), 1);
}

TEST(HostMetricsTest, FirstSampleHasNoCpuPercent) {
  const HostSample first = HostMetrics().Sample(HostSample());
  EXPECT_DOUBLE_EQ(HostMetrics::kUnavailable, first.cpu_percent);
}

// Everything below is either a real reading or the unavailable sentinel --
// never a plausible-looking zero, which would be indistinguishable from a
// genuine measurement on the page.
TEST(HostMetricsTest, ResidentMemoryIsPositiveOrUnavailable) {
  const HostSample s = HostMetrics().Sample(HostSample());
  if (HostMetrics().IsSupported()) {
    EXPECT_GT(s.rss_mb, 0.0) << "a running process occupies memory";
  } else {
    EXPECT_DOUBLE_EQ(HostMetrics::kUnavailable, s.rss_mb);
  }
}

TEST(HostMetricsTest, TemperatureIsPlausibleOrUnavailable) {
  const HostSample s = HostMetrics().Sample(HostSample());
  if (s.temp_c != HostMetrics::kUnavailable) {
    EXPECT_GT(s.temp_c, -40.0);
    EXPECT_LT(s.temp_c, 150.0);
  }
}

TEST(HostMetricsTest, CountersAdvanceAcrossSamples) {
  const HostMetrics metrics;
  const HostSample first = metrics.Sample(HostSample());
  const HostSample second = metrics.Sample(first);
  if (metrics.IsSupported()) {
    EXPECT_GE(second.cpu_seconds, first.cpu_seconds);
    EXPECT_GE(second.wall_seconds, first.wall_seconds);
  }
}

TEST(HostMetricsTest, SecondSampleReportsBoundedCpu) {
  const HostMetrics metrics;
  const HostSample first = metrics.Sample(HostSample());
  // Burn a little CPU so the interval is not degenerate.
  volatile double sink = 0.0;
  for (int i = 0; i < 2000000; ++i) { sink += static_cast<double>(i) * 0.5; }
  (void) sink;  // To suppress unused variable warning in release builds.
  const HostSample second = metrics.Sample(first);
  if (metrics.IsSupported() &&
      second.cpu_percent != HostMetrics::kUnavailable) {
    EXPECT_GE(second.cpu_percent, 0.0);
    EXPECT_LE(second.cpu_percent, 100.0);
  }
}

}  // namespace
