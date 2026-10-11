// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file server.cc
 * @brief Server implementation, backed by cpp-httplib.
 *
 * cpp-httplib is kept behind a pimpl so its header never leaks into the rest
 * of the app and the public server.hpp stays dependency-free.
 */

#include "app/server.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "httplib.h"  // NOLINT(build/include_subdir)

namespace mep_cctv {
namespace app {

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Multipart boundary for the MJPEG stream.
const char* const kBoundary = "mepcctvframe";

// Pace the stream so a viewer cannot spin the CPU re-sending identical
// frames faster than the camera produces them.
constexpr int kStreamPollMs = 10;
constexpr int kStreamMaxWaitMs = 2000;

/**
 * @brief Set same-origin content, framing, MIME, and referrer policies.
 */
int ApplySecurityHeaders(httplib::Response* res) {
  res->set_header(
      "Content-Security-Policy",
      "default-src 'self'; img-src 'self' data:; "
      "style-src 'self'; script-src 'self'; frame-ancestors 'none'");
  res->set_header("X-Content-Type-Options", "nosniff");
  res->set_header("X-Frame-Options", "DENY");
  res->set_header("Referrer-Policy", "no-referrer");
  return 4;
}

/** @brief Escape the few characters that would break a JSON string literal. */
std::string JsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (const char c : in) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

class Server::Impl {
 public:
  Impl(Pipeline* pipeline, const ServerConfig& config)
      : pipeline_(pipeline), config_(config) {
    RegisterRoutes();
  }

  bool Listen() {
    std::printf("mep-cctv listening on http://%s:%d/\n", config_.host.c_str(),
                config_.port);
    if (config_.token.empty()) {
      std::printf(
          "WARNING: no --token set; /stream and /status are "
          "unauthenticated.\n");
    }
    return server_.listen(config_.host, config_.port);
  }

  bool Stop() {
    if (!server_.is_running()) return false;

    server_.stop();
    return true;
  }

  /** @brief Append ,"key":value -- or null when the metric is unavailable. */
  int AppendMetric(std::ostringstream* os, const char* key,
                   double value) const {
    *os << R"(,")" << key << R"(:)";

    if (value < 0.0)
      *os << "null";
    else
      *os << value;
    return 1;
  }

  std::string StatusJson() const {
    const Snapshot snap = pipeline_->Latest();
    const Detector& det = pipeline_->GetDetector();
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(2);
    os << "{"
       << R"("person_present":)" << (snap.person_present ? "true" : "false")
       << R"(,"audio_score":)" << snap.audio_score << R"(,"detections":)"
       << snap.detection_count << R"(,"fps":)" << snap.fps << R"(,"sequence":)"
       << snap.sequence << R"(,"backend":")" << JsonEscape(det.BackendName())
       << '"' << R"(,"version":")" << JsonEscape(det.Version()) << '"'
       << R"(,"simd":)" << (det.SimdEnabled() ? "true" : "false")
       << R"(,"video_source":")" << JsonEscape(pipeline_->VideoSourceName())
       << '"' << R"(,"audio_source":")"
       << JsonEscape(pipeline_->AudioSourceName()) << '"';
    // JSON null distinguishes unavailable metrics from measured zero.
    AppendMetric(&os, "cpu_percent", snap.cpu_percent);
    AppendMetric(&os, "rss_mb", snap.rss_mb);
    AppendMetric(&os, "temp_c", snap.temp_c);
    os << "}";
    return os.str();
  }

 private:
  int RegisterRoutes() {
    server_.set_mount_point("/", config_.static_dir);

    server_.Get(
        "/status", [this](const httplib::Request& req, httplib::Response& res) {
          ApplySecurityHeaders(&res);
          if (!Authorized(req)) {
            res.status = 401;
            res.set_content(R"({"error":"unauthorized"})", "application/json");
            return;
          }
          res.set_content(StatusJson(), "application/json");
        });

    server_.Get(
        "/stream", [this](const httplib::Request& req, httplib::Response& res) {
          ApplySecurityHeaders(&res);
          if (!Authorized(req)) {
            res.status = 401;
            res.set_content("unauthorized", "text/plain");
            return;
          }
          if (active_streams_.load() >= config_.max_streams) {
            res.status = 503;
            res.set_content("too many concurrent streams", "text/plain");
            return;
          }
          StreamMjpeg(&res);
        });

    server_.Get("/healthz",
                [](const httplib::Request&, httplib::Response& res) {
                  res.set_content("ok", "text/plain");
                });
    return 4;
  }

  bool Authorized(const httplib::Request& req) const {
    if (config_.token.empty()) return true;

    return req.get_param_value("token") == config_.token;
  }

  int StreamMjpeg(httplib::Response* res) {
    active_streams_.fetch_add(1);
    const std::string content_type =
        std::string("multipart/x-mixed-replace; boundary=") + kBoundary;

    // last_sequence is owned by the provider closure: one per connection, so
    // each viewer independently tracks which frame it has already sent.
    auto last_sequence = std::make_shared<uint64_t>(0);

    res->set_chunked_content_provider(
        content_type,
        [this, last_sequence](size_t /*offset*/,
                              const httplib::DataSink& sink) {
          const Snapshot snap = WaitForNewFrame(last_sequence.get());
          if (snap.jpeg.empty()) {
            return true;  // nothing yet; keep the connection open
          }
          return WritePart(sink, snap);
        },
        [this](bool) { active_streams_.fetch_sub(1); });
    return 1;
  }

  /** @brief Block briefly until a frame newer than *last is available. */
  Snapshot WaitForNewFrame(uint64_t* last) const {
    for (int waited = 0; waited < kStreamMaxWaitMs; waited += kStreamPollMs) {
      const Snapshot snap = pipeline_->Latest();
      if (!snap.jpeg.empty() && snap.sequence != *last) {
        *last = snap.sequence;
        return snap;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kStreamPollMs));
    }
    return Snapshot{};
  }

  /** @brief Write one multipart chunk; false tells httplib to hang up. */
  static bool WritePart(const httplib::DataSink& sink, const Snapshot& snap) {
    std::ostringstream header;
    header << "--" << kBoundary << "\r\n"
           << "Content-Type: image/jpeg\r\n"
           << "Content-Length: " << snap.jpeg.size() << "\r\n\r\n";
    const std::string head = header.str();
    if (!sink.write(head.data(), head.size())) return false;

    if (!sink.write(reinterpret_cast<const char*>(snap.jpeg.data()),
                    snap.jpeg.size())) {
      return false;
    }

    return sink.write("\r\n", 2);
  }

  Pipeline* const pipeline_;
  const ServerConfig config_;
  std::atomic<int> active_streams_{0};
  httplib::Server server_;
};

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

Server::Server(Pipeline* pipeline, const ServerConfig& config)
    : impl_(std::make_unique<Impl>(pipeline, config)) {}

Server::~Server() = default;

bool Server::Listen() { return impl_->Listen(); }

bool Server::Stop() { return impl_->Stop(); }

std::string Server::StatusJson() const { return impl_->StatusJson(); }

}  // namespace app
}  // namespace mep_cctv
