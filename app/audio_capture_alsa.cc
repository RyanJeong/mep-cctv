// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file audio_capture_alsa.cc
 * @brief USB microphone capture via ALSA. Compiled only on Linux when
 *        libasound is available (see the CMake selection in CMakeLists.txt).
 *
 * Captures signed 16-bit mono and normalises to float in [-1, 1], which is
 * what cctv_detect_audio expects.
 */

#include <alsa/asoundlib.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "app/sources.hpp"

namespace mep_cctv {
namespace app {

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// Full-scale magnitude of a signed 16-bit sample, used to normalise to [-1,1].
constexpr float kInt16Scale = 32768.0f;

/** @brief A configured ALSA capture handle. */
class AlsaAudioSource : public AudioSource {
 public:
  AlsaAudioSource(snd_pcm_t* handle, int sample_rate, int chunk,
                  std::string device) noexcept
      : handle_(handle),
        sample_rate_(sample_rate),
        chunk_(chunk),
        device_(std::move(device)) {}

  ~AlsaAudioSource() override { snd_pcm_close(handle_); }

  std::vector<float> Read() override {
    std::vector<int16_t> raw(static_cast<std::size_t>(chunk_), 0);
    const snd_pcm_sframes_t got = snd_pcm_readi(
        handle_, raw.data(), static_cast<snd_pcm_uframes_t>(chunk_));
    if (got < 0) {
      // Recover from an overrun and report silence for this chunk rather
      // than propagating an error up through the capture loop.
      snd_pcm_recover(handle_, static_cast<int>(got), 1);
      return std::vector<float>(static_cast<std::size_t>(chunk_), 0.0f);
    }

    std::vector<float> samples(static_cast<std::size_t>(chunk_), 0.0f);
    for (snd_pcm_sframes_t i = 0; i < got; ++i) {
      samples[static_cast<std::size_t>(i)] =
          static_cast<float>(raw[static_cast<std::size_t>(i)]) / kInt16Scale;
    }
    return samples;
  }

  int SampleRate() const override { return sample_rate_; }

  std::string Name() const override { return "alsa:" + device_; }

 private:
  snd_pcm_t* const handle_;
  const int sample_rate_;
  const int chunk_;
  const std::string device_;
};

}  // namespace

std::unique_ptr<AudioSource> OpenAudioCapture(const std::string& device,
                                              int sample_rate, int chunk) {
  snd_pcm_t* handle = nullptr;
  if (snd_pcm_open(&handle, device.c_str(), SND_PCM_STREAM_CAPTURE, 0) < 0)
    return nullptr;

  const auto rate = static_cast<unsigned>(sample_rate);
  const int status = snd_pcm_set_params(
      handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
      1 /* mono */, rate, 1 /* allow resampling */, 500000 /* 0.5 s latency */);
  if (status < 0) {
    snd_pcm_close(handle);
    return nullptr;
  }

  return std::unique_ptr<AudioSource>(
      new AlsaAudioSource(handle, sample_rate, chunk, device));
}

}  // namespace app
}  // namespace mep_cctv
