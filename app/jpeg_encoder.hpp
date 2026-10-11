// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_JPEG_ENCODER_HPP_
#define APP_JPEG_ENCODER_HPP_

/**
 * @file jpeg_encoder.hpp
 * @brief Encodes an RGB frame to an in-memory JPEG.
 *
 * Output goes to a byte buffer rather than a file because each frame becomes
 * one part of the multipart/x-mixed-replace MJPEG response; nothing is ever
 * written to disk.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "app/frame.hpp"

namespace mep_cctv {
namespace app {

/** @brief RGB888 -> JPEG encoder. */
class JpegEncoder {
 public:
  /** @brief Default quality, balancing bandwidth against artefacts. */
  static const int kDefaultQuality;

  explicit JpegEncoder(int quality = 0) noexcept;

  /**
   * @brief Encode a frame to JPEG bytes.
   *
   * @param frame  Frame to encode; must be valid.
   * @return JPEG bytes, or an empty vector when the frame is invalid.
   *
   * @side_effects None -- referentially transparent: same frame in, same
   *               bytes out. Allocation is the returned value itself.
   */
  std::vector<uint8_t> Encode(const Frame& frame) const;

  /** @brief Quality this encoder was constructed with, in 1..100. */
  int Quality() const;

 private:
  const int quality_;
};

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_JPEG_ENCODER_HPP_
