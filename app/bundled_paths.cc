// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#include "app/bundled_paths.hpp"

#include <fstream>
#include <string>

namespace mep_cctv {
namespace app {
namespace {

/** @brief True when the path names a readable file. */
bool FileExists(const std::string& path) {
  const std::ifstream probe(path.c_str());
  return probe.good();
}

}  // namespace

std::string ExecutableDirectory(const char* argv0) {
  const std::string path(argv0 != nullptr ? argv0 : "");
  const std::string::size_type slash = path.find_last_of('/');
  return (slash == std::string::npos) ? std::string(".")
                                      : path.substr(0, slash);
}

std::string FindBundledModel(const char* argv0) {
  const std::string candidate =
      ExecutableDirectory(argv0) + "/" + "yolov8n.onnx";
  return FileExists(candidate) ? candidate : std::string();
}

std::string FindBundledStaticDir(const char* argv0) {
  const std::string candidate = ExecutableDirectory(argv0) + "/" + "static";
  return FileExists(candidate + "/index.html") ? candidate : std::string();
}

}  // namespace app
}  // namespace mep_cctv
