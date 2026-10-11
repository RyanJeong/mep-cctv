// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_HOST_METRICS_HPP_
#define APP_HOST_METRICS_HPP_

/**
 * @file host_metrics.hpp
 * @brief Linux process CPU, memory, and CPU-temperature measurements.
 *
 * Reads /proc and /sys. Unsupported metrics use kUnavailable rather than zero.
 */

namespace mep_cctv {
namespace app {

/**
 * @brief One reading, plus the raw counters needed to take the next one.
 *
 * Pass the previous sample back to compute CPU usage over an interval.
 */
struct HostSample {
  double cpu_percent = -1.0;  /**< 0-100 across all cores; -1 unavailable */
  double rss_mb = -1.0;       /**< Resident set size in MiB; -1 unavailable */
  double temp_c = -1.0;       /**< SoC temperature in Celsius; -1 unavail. */
  double cpu_seconds = -1.0;  /**< Cumulative process CPU time consumed */
  double wall_seconds = -1.0; /**< Monotonic clock reading of this sample */
};

/** @brief Reads process and board metrics. */
class HostMetrics {
 public:
  /** @brief Sentinel for a metric this platform cannot provide. */
  static const double kUnavailable;

  /** @brief True when this build can read the metrics at all. */
  bool IsSupported() const;

  /**
   * @brief Take a reading.
   *
   * @param previous The last sample, or a default-constructed HostSample on
   *                 the first call. cpu_percent is kUnavailable until two
   *                 readings exist, because a rate needs an interval.
   * @return The new sample.
   * @note Reads /proc and /sys; the result depends on machine state.
   */
  HostSample Sample(const HostSample& previous) const;

  /** @brief Number of online CPUs, or 1 when that cannot be determined. */
  int CoreCount() const;
};

/**
 * @brief CPU used between two samples, as a percentage of total capacity.
 *
 * 100 means every core was busy for the whole interval, which is what makes
 * the number comparable across a 4-core Pi and a laptop. Pure arithmetic, no
 * I/O.
 *
 * @param previous Earlier sample.
 * @param current  Later sample.
 * @param cores    Online CPUs; values below 1 are treated as 1.
 * @return Percentage clamped to 0-100, or kUnavailable when either sample
 *         lacks counters or no time elapsed between them.
 */
double CpuPercentBetween(const HostSample& previous, const HostSample& current,
                         int cores);

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_HOST_METRICS_HPP_
