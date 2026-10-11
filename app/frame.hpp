// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_FRAME_HPP_
#define APP_FRAME_HPP_

/**
 * @file frame.hpp
 * @brief Plain 8-bit RGB image buffer passed between capture, inference,
 *        annotation and encoding.
 *
 * Row-major, three bytes per pixel, no padding between rows -- exactly the
 * layout cctv_detect_objects expects, so a Frame can be handed to the C ABI
 * without a copy or a stride fix-up.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mep_cctv {
namespace app {

/** @brief An RGB888 image. Empty when width or height is zero. */
struct Frame {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;

  /** @brief Return true when the buffer size matches the declared geometry. */
  bool IsValid() const {
    return width > 0 && height > 0 &&
           rgb.size() == static_cast<std::size_t>(width) *
                             static_cast<std::size_t>(height) * 3u;
  }
};

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_FRAME_HPP_
