// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef TESTS_APP_FRAME_HELPERS_HPP_
#define TESTS_APP_FRAME_HELPERS_HPP_

/**
 * @file frame_helpers.hpp
 * @brief Frame and box builders shared by the app test files.
 *
 * Several units under test take a Frame or a cctv_box, so these two builders
 * would otherwise be copied into every file that needs one.
 */

#include <cstddef>
#include <cstdint>

#include "app/frame.hpp"
#include "cctv/cctv.h"

namespace mep_cctv {
namespace app {
namespace testing {

/** @brief Build a solid mid-grey frame. */
inline Frame MakeFrame(int w, int h) {
  Frame f;
  f.width = w;
  f.height = h;
  f.rgb.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u,
               128u);
  return f;
}

/** @brief Build a detection box with a plausible score and class. */
inline cctv_box MakeBox(float x, float y, float w, float h) {
  cctv_box b;
  b.x = x;
  b.y = y;
  b.w = w;
  b.h = h;
  b.score = 0.9f;
  b.class_id = 0;
  return b;
}

}  // namespace testing
}  // namespace app
}  // namespace mep_cctv

#endif  // TESTS_APP_FRAME_HELPERS_HPP_
