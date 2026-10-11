// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef SRC_CORE_AUDIO_DETECT_HPP_
#define SRC_CORE_AUDIO_DETECT_HPP_

/**
 * @file audio_detect.hpp
 * @brief Code-based acoustic activity detection (AAD).
 *
 * Answers one question: "is something making noise near the camera?"
 *
 * The detector is short-term energy (RMS) against a threshold, with the DC
 * offset removed first -- cheap microphones often sit on a constant bias, and
 * raw RMS would report that silent bias as sound.
 */

#include "cctv/cctv.h"

namespace mep_cctv {
namespace core {

/** @brief RMS at or above this level counts as sound. */
extern const float kAudioRmsThreshold;

/**
 * @brief Detect acoustic activity in a block of mono PCM.
 *
 * @param samples      Mono PCM float samples, nominally in [-1, 1].
 *                     May be NULL only when @p num_samples is 0.
 * @param num_samples  Number of samples; values <= 0 yield a zero result.
 * @return cctv_audio_result whose @c person_present is 1 when the
 *         DC-corrected RMS reaches kAudioRmsThreshold, and whose @c score is
 *         that RMS clamped to [0, 1].
 *
 * @side_effects None -- referentially transparent: same samples in, same
 *               result out. No allocation, no I/O, no state.
 */
cctv_audio_result DetectAcousticActivity(const float* samples,
                                         int num_samples) noexcept;

}  // namespace core
}  // namespace mep_cctv

#endif  // SRC_CORE_AUDIO_DETECT_HPP_
