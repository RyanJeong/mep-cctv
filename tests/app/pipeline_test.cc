// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file pipeline_test.cc
 * @brief Tests for app/pipeline.cc.
 *
 * Native-only tests need no external ONNX model path.
 */

#include "app/pipeline.hpp"
#include "app/server.hpp"

#include <chrono>
#include <thread>

#include "gtest/gtest.h"

namespace {

using mep_cctv::app::Pipeline;
using mep_cctv::app::PipelineConfig;

#if defined(CCTV_USE_NATIVE_BACKEND) && CCTV_USE_NATIVE_BACKEND
TEST(PipelineTest, ProducesASnapshotWithSyntheticSources) {
  PipelineConfig cfg;
  cfg.width = 160;
  cfg.height = 120;
  cfg.audio_chunk = 1600;
  cfg.video_device = "/nonexistent";  // force the synthetic fallback
  Pipeline pipeline(cfg);
  ASSERT_TRUE(pipeline.GetDetector().IsReady());
  ASSERT_TRUE(pipeline.Start());

  // Wait for the producer thread with a bounded deadline: poll on a short
  // sleep rather than spinning, so the test does not depend on how fast this
  // machine happens to reach the first frame.
  //
  // The budget depends on which model is linked. The placeholder returns a
  // fixed tensor immediately, but the real TVM-compiled network is
  // unscheduled and needs seconds per frame -- tens of seconds on a slow or
  // emulated machine -- so a five-second deadline would fail there for a
  // reason that has nothing to do with correctness.
#if defined(CCTV_USE_GENERATED_MODEL) && CCTV_USE_GENERATED_MODEL
  const int kMaxPolls = 4800;  // 120 s
#else
  const int kMaxPolls = 200;  // 5 s
#endif
  mep_cctv::app::Snapshot snap;
  for (int i = 0; i < kMaxPolls && snap.jpeg.empty(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    snap = pipeline.Latest();
  }
  pipeline.Stop();
  EXPECT_FALSE(snap.jpeg.empty());
  EXPECT_GT(snap.sequence, 0u);
}

TEST(PipelineTest, FallsBackToSyntheticWhenNoCamera) {
  PipelineConfig cfg;
  cfg.video_device = "/nonexistent";
  const Pipeline pipeline(cfg);
  EXPECT_EQ("synthetic", pipeline.VideoSourceName());
}

TEST(PipelineTest, StopWithoutStartReportsFalse) {
  PipelineConfig cfg;
  Pipeline pipeline(cfg);
  EXPECT_FALSE(pipeline.Stop());
}

TEST(ServerStatusTest, UnavailableMetricsHaveQuotedKeysAndNullValues) {
  PipelineConfig cfg;
  cfg.video_device = "/nonexistent";
  cfg.audio_device = "/nonexistent";
  Pipeline pipeline(cfg);
  const mep_cctv::app::Server server(
      &pipeline, mep_cctv::app::ServerConfig{});

  const std::string json = server.StatusJson();
  EXPECT_NE(std::string::npos, json.find(R"("detections":0)"));
  EXPECT_NE(std::string::npos, json.find(R"("fps":0.00)"));
  EXPECT_NE(std::string::npos,
            json.find(R"(,"cpu_percent":null,"rss_mb":null,"temp_c":null})"));
}
#endif  // CCTV_USE_NATIVE_BACKEND

}  // namespace
