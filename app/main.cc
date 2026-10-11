// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file main.cc
 * @brief mep-cctv application entry point.
 *
 * Starts the capture/inference pipeline and serves the monitor page.
 * Open http://<host>:8080/ in a browser to watch the annotated stream.
 */

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "app/bundled_paths.hpp"
#include "app/pipeline.hpp"
#include "app/server.hpp"

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// The signal handler marks shutdown and requests a stop on the active server.
// Server::Stop is not guaranteed async-signal-safe.
mep_cctv::app::Server* g_server = nullptr;  // NOSONAR - set while serving.
volatile std::sig_atomic_t g_stop_requested = 0;  // NOSONAR - signal flag.

void HandleSignal(int /*signum*/) {
  g_stop_requested = 1;
  if (g_server != nullptr) g_server->Stop();
}

/** @brief Read an option of the form --name value; returns true if consumed. */
bool MatchOption(const char* arg, const char* next, const char* name,
                 std::string* out) {
  if (std::string(arg) != name || next == nullptr) return false;

  *out = next;
  return true;
}

bool MatchIntegerOption(const char* arg, const char* next, const char* name,
                        int* out) {
  std::string value;
  if (!MatchOption(arg, next, name, &value)) return false;

  *out = std::atoi(value.c_str());
  return true;
}

/** @brief Print usage. Returns the exit status to use. */
int PrintUsage() {
  std::printf(
      "mep-cctv -- tiny CCTV with object detection\n"
      "\n"
      "Usage: cctv_app [OPTIONS]\n"
      "\n"
      "Options:\n"
      "  --host <addr>       Bind address (default 0.0.0.0)\n"
      "  --port <n>          Port (default 8080)\n"
      "  --video <dev>       V4L2 device (default /dev/video0)\n"
      "  --audio <dev>       ALSA device (default \"default\")\n"
      "  --width <n>         Capture width (default 640)\n"
      "  --height <n>        Capture height (default 480)\n"
      "  --yolo <path>       YOLO .onnx (onnx mode; defaults to\n"
      "                      yolov8n.onnx beside the executable)\n"
      "  --static <dir>      Static asset directory (defaults to static/\n"
      "                      beside the executable, else app/static)\n"
      "  --token <secret>    Require ?token=<secret> on /stream and /status\n"
      "  --quality <1-100>   JPEG quality (default 80)\n"
      "  -h, --help          Show this help\n"
      "\n"
      "The native mode has the model compiled in and ignores --yolo.\n"
      "The onnx mode requires it.\n"
      "\n"
      "Audio is a plain energy detector, not a model -- no audio weights\n"
      "are needed in any mode.\n");
  return 0;
}

enum class ArgumentResult {
  kContinue,
  kHelp,
  kError
};

ArgumentResult ParseArguments(int argc, char** argv,
                              mep_cctv::app::PipelineConfig* pipeline_config,
                              mep_cctv::app::ServerConfig* server_config) {
  int index = 1;
  while (index < argc) {
    const char* const arg = argv[index];
    const char* const next = (index + 1 < argc) ? argv[index + 1] : nullptr;

    if (std::string(arg) == "-h" || std::string(arg) == "--help")
      return ArgumentResult::kHelp;

    if (MatchOption(arg, next, "--host", &server_config->host) ||
        MatchOption(arg, next, "--static", &server_config->static_dir) ||
        MatchOption(arg, next, "--token", &server_config->token) ||
        MatchOption(arg, next, "--video", &pipeline_config->video_device) ||
        MatchOption(arg, next, "--audio", &pipeline_config->audio_device) ||
        MatchOption(arg, next, "--yolo", &pipeline_config->yolo_model)) {
      index += 2;
      continue;
    }
    if (MatchIntegerOption(arg, next, "--port", &server_config->port) ||
        MatchIntegerOption(arg, next, "--width", &pipeline_config->width) ||
        MatchIntegerOption(arg, next, "--height", &pipeline_config->height) ||
        MatchIntegerOption(arg, next, "--quality",
                           &pipeline_config->jpeg_quality)) {
      index += 2;
      continue;
    }

    std::fprintf(stderr, "unknown option: %s\n", arg);
    return ArgumentResult::kError;
  }
  return ArgumentResult::kContinue;
}

}  // namespace

int main(int argc, char** argv) {
  // Flush complete log lines when stdout is redirected to a service journal.
  setvbuf(stdout, nullptr, _IOLBF, 0);

  mep_cctv::app::PipelineConfig pipeline_config;
  mep_cctv::app::ServerConfig server_config;

  // Asset discovery runs when the configured path equals the default.
  const std::string default_static_dir = server_config.static_dir;

  const ArgumentResult argument_result =
      ParseArguments(argc, argv, &pipeline_config, &server_config);
  if (argument_result == ArgumentResult::kHelp) return PrintUsage();
  if (argument_result == ArgumentResult::kError) return PrintUsage() + 1;

  // Fall back to the model shipped in the artifact folder. Harmless for the
  // native mode, which ignores the path entirely.
  if (pipeline_config.yolo_model.empty()) {
    pipeline_config.yolo_model = mep_cctv::app::FindBundledModel(argv[0]);
    if (!pipeline_config.yolo_model.empty()) {
      std::printf("model        : %s (found beside the executable)\n",
                  pipeline_config.yolo_model.c_str());
    }
  }

  // Prefer executable-side web assets over the repository-relative default.
  if (server_config.static_dir == default_static_dir) {
    const std::string bundled = mep_cctv::app::FindBundledStaticDir(argv[0]);
    if (!bundled.empty()) server_config.static_dir = bundled;
  }

  mep_cctv::app::Pipeline pipeline(pipeline_config);
  if (!pipeline.GetDetector().IsReady()) {
    std::fprintf(stderr,
                 "failed to create the inference engine.\n"
                 "The onnx mode needs a YOLO model: pass --yolo <path>, or "
                 "put yolov8n.onnx next to cctv_app.\n");
    return 1;
  }

  if (!pipeline.Start()) {
    std::fprintf(stderr, "failed to start the capture pipeline\n");
    return 1;
  }

  std::printf("backend      : %s (simd=%s)\n",
              pipeline.GetDetector().BackendName().c_str(),
              pipeline.GetDetector().SimdEnabled() ? "on" : "off");
  std::printf("video source : %s\n", pipeline.VideoSourceName().c_str());
  std::printf("audio source : %s\n", pipeline.AudioSourceName().c_str());

  mep_cctv::app::Server server(&pipeline, server_config);
  g_server = &server;
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  const bool ok = server.Listen();
  g_server = nullptr;
  pipeline.Stop();

  if (!ok && g_stop_requested == 0) {
    std::fprintf(stderr, "failed to bind %s:%d\n", server_config.host.c_str(),
                 server_config.port);
    return 1;
  }
  std::printf("stopped\n");
  return 0;
}
