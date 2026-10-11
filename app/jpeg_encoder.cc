// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file jpeg_encoder.cc
 * @brief JpegEncoder implementation, backed by stb_image_write.
 *
 * stb_image_write is a single-header, public-domain encoder. It is used here
 * instead of libjpeg because it adds no system dependency to the Raspberry Pi
 * image and keeps the aarch64 cross-build in docker/Dockerfile.crossbuild a
 * plain compile with no extra sysroot libraries.
 */

#include "app/jpeg_encoder.hpp"

#include <algorithm>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO   // in-memory only; never touch the filesystem
#include "stb_image_write.h"  // NOLINT(build/include_subdir)

namespace mep_cctv {
namespace app {

const int JpegEncoder::kDefaultQuality = 80;

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

constexpr int kMinQuality = 1;
constexpr int kMaxQuality = 100;
constexpr int kRgbChannels = 3;

/** @brief stb write callback: appends encoded bytes to a std::vector. */
void AppendToVector(void* context, void* data, int size) {
  auto* out = static_cast<std::vector<uint8_t>*>(context);
  const auto* bytes = static_cast<const uint8_t*>(data);
  out->insert(out->end(), bytes, bytes + size);
}

}  // namespace

JpegEncoder::JpegEncoder(int quality) noexcept
    : quality_(
          quality <= 0
              ? kDefaultQuality
              : std::min(kMaxQuality,  // NOLINT(whitespace/indent_namespace)
                         std::max(kMinQuality, quality))) {}

std::vector<uint8_t> JpegEncoder::Encode(const Frame& frame) const {
  std::vector<uint8_t> out;
  if (!frame.IsValid()) return out;

  const int ok =
      stbi_write_jpg_to_func(&AppendToVector, &out, frame.width, frame.height,
                             kRgbChannels, frame.rgb.data(), quality_);
  if (ok == 0) out.clear();
  return out;
}

int JpegEncoder::Quality() const { return quality_; }

}  // namespace app
}  // namespace mep_cctv
