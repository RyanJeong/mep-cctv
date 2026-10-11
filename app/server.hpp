// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_SERVER_HPP_
#define APP_SERVER_HPP_

/**
 * @file server.hpp
 * @brief HTTP monitor, MJPEG stream, and status endpoint.
 *
 * Routes:
 *   GET /          the monitor page
 *   GET /stream    live MJPEG (multipart/x-mixed-replace) with boxes burned in
 *   GET /status    JSON: sound activity, detection count, fps, backend, metrics
 *   GET /healthz   liveness response
 *   GET /style.css, /app.js   static assets
 *
 * Annotated MJPEG frames render directly in the browser's img element.
 */

#include <memory>
#include <string>

#include "app/pipeline.hpp"

namespace mep_cctv {
namespace app {

/** @brief Server configuration. */
struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = 8080;
  std::string static_dir = "app/static";
  // Protects /stream and /status only; empty means unauthenticated access.
  std::string token;
  // Stream admission limit; requests check the current active count.
  int max_streams = 4;
};

/** @brief Serves the CCTV monitor over HTTP. */
class Server {
 public:
  Server(Pipeline* pipeline, const ServerConfig& config);
  ~Server();

  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  /**
   * @brief Bind the port and serve until Stop() is called.
   *
   * @return True on a clean shutdown, false when the port could not be bound.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Binds a listening socket and blocks serving clients.
   * @side_effects_what   Network I/O; reads pipeline snapshots.
   * @side_effects_impact Occupies the calling thread until shutdown.
   * @side_effects_alternatives None -- serving is inherently effectful.
   */
  bool Listen();

  /**
   * @brief Stop a running server.
   *
   * @return True when a running server was asked to stop.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Closes the listening socket.
   * @side_effects_what   Network teardown.
   * @side_effects_impact Listen() returns shortly afterwards.
   * @side_effects_alternatives None.
   */
  bool Stop();

  /** @brief Render the /status payload for the current snapshot. */
  std::string StatusJson() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_SERVER_HPP_
