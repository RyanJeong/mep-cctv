// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file video_capture_v4l2.cc
 * @brief USB webcam capture via Video4Linux2. Compiled on Linux only.
 *
 * CMake selects this file for Linux targets and video_capture_null.cc
 * everywhere else, so sources.hpp needs no #ifdef.
 *
 * Pipeline: negotiate YUYV (the format essentially every UVC webcam offers),
 * stream through mmap'd buffers, and convert each dequeued buffer to RGB888.
 * We deliberately do NOT request the camera's MJPEG mode: the frames would
 * have to be decoded again before boxes could be drawn on them.
 */

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "app/sources.hpp"

namespace mep_cctv {
namespace app {

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Number of mmap'd capture buffers. Four is enough to avoid stalling the
// driver while one frame is being processed.
constexpr int kBufferCount = 4;

/** @brief Retry an ioctl across EINTR; return the final result. */
template <typename T>
int RetryIoctl(int fd, unsigned long request, T* arg) {  // NOLINT
  int result = 0;
  do {
    result = ioctl(fd, request, arg);
  } while (result == -1 && errno == EINTR);
  return result;
}

/** @brief Clamp an int into the 0..255 byte range. */
uint8_t ClampByte(int value) {
  return static_cast<uint8_t>(std::min(255, std::max(0, value)));
}

/** @brief One mmap'd V4L2 buffer. */
struct MappedBuffer {
  void* start = nullptr;
  std::size_t length = 0;
};

/** @brief A streaming V4L2 capture device. */
class V4l2VideoSource : public VideoSource {
 public:
  V4l2VideoSource(int fd, int width, int height, std::string device,
                  std::vector<MappedBuffer> buffers) noexcept
      : fd_(fd),
        width_(width),
        height_(height),
        device_(std::move(device)),
        buffers_(std::move(buffers)) {}

  V4l2VideoSource(const V4l2VideoSource&) = delete;
  V4l2VideoSource& operator=(const V4l2VideoSource&) = delete;

  ~V4l2VideoSource() override {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    RetryIoctl(fd_, VIDIOC_STREAMOFF, &type);
    for (MappedBuffer& b : buffers_) {
      if (b.start != nullptr && b.start != MAP_FAILED) {
        munmap(b.start, b.length);
      }
    }
    close(fd_);
  }

  Frame Read() override {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (RetryIoctl(fd_, VIDIOC_DQBUF, &buf) == -1) return Frame{};

    Frame frame = ConvertYuyv(buf);
    RetryIoctl(fd_, VIDIOC_QBUF, &buf);
    return frame;
  }

  std::string Name() const override { return "v4l2:" + device_; }

 private:
  /** @brief Convert the dequeued YUYV buffer into an RGB888 Frame. */
  Frame ConvertYuyv(const v4l2_buffer& buf) const {
    Frame frame;
    frame.width = width_;
    frame.height = height_;
    frame.rgb.assign(static_cast<std::size_t>(width_) *
                         static_cast<std::size_t>(height_) * 3u,
                     0u);

    const std::size_t needed = static_cast<std::size_t>(width_) *
                               static_cast<std::size_t>(height_) * 2u;
    if (buf.index >= buffers_.size() || buffers_[buf.index].length < needed)
      return frame;

    auto src = static_cast<const uint8_t*>(buffers_[buf.index].start);

    // YUYV packs two pixels into four bytes: Y0 Cb Y1 Cr.
    const int pair_count = width_ * height_ / 2;
    for (int p = 0; p < pair_count; ++p) {
      const int si = p * 4;
      const int y0 = src[si];
      const int cb = src[si + 1];
      const int y1 = src[si + 2];
      const int cr = src[si + 3];
      WritePixel(&frame.rgb, static_cast<std::size_t>(p) * 6u, y0, cb, cr);
      WritePixel(&frame.rgb, static_cast<std::size_t>(p) * 6u + 3u, y1, cb, cr);
    }
    return frame;
  }

  /** @brief BT.601 YCbCr -> RGB for one pixel; returns bytes written. */
  static int WritePixel(std::vector<uint8_t>* rgb, std::size_t at, int y,
                        int cb, int cr) {
    const int c = y - 16;
    const int d = cb - 128;
    const int e = cr - 128;
    (*rgb)[at] = ClampByte((298 * c + 409 * e + 128) >> 8);
    (*rgb)[at + 1] = ClampByte((298 * c - 100 * d - 208 * e + 128) >> 8);
    (*rgb)[at + 2] = ClampByte((298 * c + 516 * d + 128) >> 8);
    return 3;
  }

  const int fd_;
  const int width_;
  const int height_;
  const std::string device_;
  std::vector<MappedBuffer> buffers_;
};

/** @brief Negotiate YUYV at the requested geometry; return true on success. */
bool NegotiateFormat(int fd, int* width, int* height) {
  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = static_cast<unsigned>(*width);
  fmt.fmt.pix.height = static_cast<unsigned>(*height);
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;

  if (RetryIoctl(fd, VIDIOC_S_FMT, &fmt) == -1) return false;

  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) return false;

  // The driver may have adjusted the geometry; honour what it chose.
  *width = static_cast<int>(fmt.fmt.pix.width);
  *height = static_cast<int>(fmt.fmt.pix.height);
  return true;
}

/** @brief Request, map and enqueue the capture buffers. */
bool SetUpBuffers(int fd, std::vector<MappedBuffer>* buffers) {
  v4l2_requestbuffers req{};
  req.count = kBufferCount;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (RetryIoctl(fd, VIDIOC_REQBUFS, &req) == -1 || req.count == 0)
    return false;

  for (unsigned i = 0; i < req.count; ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    if (RetryIoctl(fd, VIDIOC_QUERYBUF, &buf) == -1) return false;

    MappedBuffer mapped;
    mapped.length = buf.length;
    mapped.start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED,
                        fd, static_cast<off_t>(buf.m.offset));
    if (mapped.start == MAP_FAILED) return false;

    buffers->push_back(mapped);
    if (RetryIoctl(fd, VIDIOC_QBUF, &buf) == -1) return false;
  }
  return true;
}

}  // namespace

std::unique_ptr<VideoSource> OpenVideoCapture(const std::string& device,
                                              int width, int height) {
  const int fd = open(device.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (fd == -1) return nullptr;

  v4l2_capability cap{};
  if (RetryIoctl(fd, VIDIOC_QUERYCAP, &cap) == -1 ||
      (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
      (cap.capabilities & V4L2_CAP_STREAMING) == 0) {
    close(fd);
    return nullptr;
  }

  int negotiated_w = width;
  int negotiated_h = height;
  if (!NegotiateFormat(fd, &negotiated_w, &negotiated_h)) {
    close(fd);
    return nullptr;
  }

  std::vector<MappedBuffer> buffers;
  if (!SetUpBuffers(fd, &buffers)) {
    for (MappedBuffer& b : buffers) {
      if (b.start != nullptr && b.start != MAP_FAILED)
        munmap(b.start, b.length);
    }
    close(fd);
    return nullptr;
  }

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (RetryIoctl(fd, VIDIOC_STREAMON, &type) == -1) {
    close(fd);
    return nullptr;
  }

  // Blocking reads from here on: the caller wants the next frame, not EAGAIN.
  const int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);

  return std::make_unique<V4l2VideoSource>(fd, negotiated_w, negotiated_h,
                                           device, std::move(buffers));
}

}  // namespace app
}  // namespace mep_cctv
