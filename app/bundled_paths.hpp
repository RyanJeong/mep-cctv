// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file bundled_paths.hpp
 * @brief Locate assets that ship alongside the executable.
 *
 * Paths are based on the directory component of argv[0], not a PATH lookup.
 * Launch copied artifacts with an executable path such as ./cctv_app.
 */

#ifndef APP_BUNDLED_PATHS_HPP_
#define APP_BUNDLED_PATHS_HPP_

#include <string>

namespace mep_cctv {
namespace app {

/**
 * @brief Directory part of argv[0].
 *
 * @param argv0 argv[0] as received by main; may be null.
 * @return The directory, or "." when argv0 has no directory component.
 */
std::string ExecutableDirectory(const char* argv0);

/**
 * @brief Locate the YOLO model shipped alongside the executable.
 *
 * Finding it automatically means the documented "copy the folder and run"
 * flow works without repeating the path on the command line; --yolo still
 * overrides.
 *
 * @param argv0 argv[0] as received by main.
 * @return Path to the model, or an empty string when none is present.
 * @note Probes the filesystem; the result depends on what is on disk.
 */
std::string FindBundledModel(const char* argv0);

/**
 * @brief Locate the static web assets shipped alongside the executable.
 *
 * A candidate directory must contain a readable index.html.
 *
 * @param argv0 argv[0] as received by main.
 * @return Path to the directory, or an empty string when none is present.
 * @note Probes the filesystem; the result depends on what is on disk.
 */
std::string FindBundledStaticDir(const char* argv0);

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_BUNDLED_PATHS_HPP_
