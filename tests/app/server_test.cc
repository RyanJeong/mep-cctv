// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#include "app/server.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include "httplib.h"
#include "gtest/gtest.h"

namespace {

#if defined(CCTV_USE_NATIVE_BACKEND) && CCTV_USE_NATIVE_BACKEND
class ServerTest : public ::testing::Test {
protected:
  void SetUp() override {
    mep_cctv::app::PipelineConfig config;
    config.video_device = "/nonexistent";
    config.audio_device = "/nonexistent";
    config.width = 32;
    config.height = 32;
    pipeline_.reset(new mep_cctv::app::Pipeline(config));
  }

  void TearDown() override {
    if (server_)
      server_->Stop();
    if (worker_.joinable())
      worker_.join();
    pipeline_->Stop();
  }

  bool Start(const std::string &token, int max_streams = 4) {
    std::random_device random;
    for (int attempt = 0; attempt < 10; ++attempt) {
      mep_cctv::app::ServerConfig config;
      config.host = "127.0.0.1";
      config.port = 20000 + static_cast<int>(random() % 30000);
      config.static_dir = "/nonexistent";
      config.token = token;
      config.max_streams = max_streams;
      server_.reset(new mep_cctv::app::Server(pipeline_.get(), config));
      finished_ = false;
      worker_ = std::thread([this] {
        server_->Listen();
        finished_ = true;
      });
      client_.reset(new httplib::Client(config.host, config.port));
      client_->set_connection_timeout(0, 100000);
      client_->set_read_timeout(3);
      for (int poll = 0; poll < 100 && !finished_; ++poll) {
        const auto response = client_->Get("/healthz");
        if (response && response->status == 200)
          return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      server_->Stop();
      worker_.join();
    }
    return false;
  }

  std::unique_ptr<mep_cctv::app::Pipeline> pipeline_;
  std::unique_ptr<mep_cctv::app::Server> server_;
  std::unique_ptr<httplib::Client> client_;
  std::atomic<bool> finished_{false};
  std::thread worker_;
};

TEST_F(ServerTest, PublicStatusReportsMetricsAndSecurityHeaders) {
  ASSERT_TRUE(pipeline_->Start());
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  ASSERT_TRUE(Start(""));
  const auto response = client_->Get("/status");
  ASSERT_TRUE(response);
  EXPECT_EQ(200, response->status);
  EXPECT_EQ("application/json", response->get_header_value("Content-Type"));
  EXPECT_EQ("nosniff", response->get_header_value("X-Content-Type-Options"));
  EXPECT_EQ("DENY", response->get_header_value("X-Frame-Options"));
  EXPECT_EQ("no-referrer", response->get_header_value("Referrer-Policy"));
  EXPECT_FALSE(response->get_header_value("Content-Security-Policy").empty());
  EXPECT_NE(std::string::npos, response->body.find(R"("backend":"native")"));
  EXPECT_NE(std::string::npos, response->body.find(R"("cpu_percent":)"));
}

TEST_F(ServerTest, TokenProtectsStatusAndStreamButNotHealth) {
  ASSERT_TRUE(Start("test-token"));
  for (const char* path :
       {"/status", "/stream", "/status?token=wrong"}) {
    const auto response = client_->Get(path);
    ASSERT_TRUE(response);
    EXPECT_EQ(401, response->status);
  }
  const auto status = client_->Get("/status?token=test-token");
  ASSERT_TRUE(status);
  EXPECT_EQ(200, status->status);
  const auto health = client_->Get("/healthz");
  ASSERT_TRUE(health);
  EXPECT_EQ("ok", health->body);
}

TEST_F(ServerTest, StreamLimitRejectsRequests) {
  ASSERT_TRUE(Start("", 0));
  const auto response = client_->Get("/stream");
  ASSERT_TRUE(response);
  EXPECT_EQ(503, response->status);
  EXPECT_EQ("too many concurrent streams", response->body);
}

TEST_F(ServerTest, StreamContainsMultipartJpegAndReleasesConnection) {
  ASSERT_TRUE(pipeline_->Start());
  ASSERT_TRUE(Start("test-token"));
  std::string received;
  client_->Get("/stream?token=test-token",
               [&received](const char *data, size_t size) {
                 received.append(data, size);
                 return received.find("\xff\xd8") == std::string::npos;
               });
  EXPECT_NE(std::string::npos, received.find("--mepcctvframe\r\n"));
  EXPECT_NE(std::string::npos, received.find("Content-Type: image/jpeg"));
  EXPECT_NE(std::string::npos, received.find("Content-Length:"));
  EXPECT_NE(std::string::npos, received.find("\xff\xd8"));
}

TEST_F(ServerTest, StreamWithoutFramesTimesOutWithoutFabricatedJpeg) {
  ASSERT_TRUE(Start(""));
  client_->set_read_timeout(3);
  std::string received;
  const auto response =
      client_->Get("/stream", [&received](const char *data, size_t size) {
        received.append(data, size);
        return true;
      });
  EXPECT_FALSE(response);
  EXPECT_TRUE(received.empty());
}

TEST(ServerLifecycleTest, StopBeforeListenAndInvalidBindReportFailure) {
  mep_cctv::app::PipelineConfig pipeline_config;
  pipeline_config.video_device = "/nonexistent";
  pipeline_config.audio_device = "/nonexistent";
  mep_cctv::app::Pipeline pipeline(pipeline_config);
  mep_cctv::app::ServerConfig config;
  config.host = "invalid bind address";
  mep_cctv::app::Server server(&pipeline, config);
  EXPECT_FALSE(server.Stop());
  EXPECT_FALSE(server.Listen());
}
#endif

} // namespace
