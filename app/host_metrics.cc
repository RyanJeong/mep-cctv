// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#include "app/host_metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace mep_cctv {
namespace app {

const double HostMetrics::kUnavailable = -1.0;

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

/** @brief Monotonic seconds; never jumps when the wall clock is adjusted. */
double NowSeconds() {
  const std::chrono::steady_clock::duration since =
      std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::duration<double>>(since)
      .count();
}

#if defined(__linux__)

/**
 * @brief Cumulative CPU seconds this process has used.
 *
 * Fields 14 and 15 of /proc/self/stat are utime and stime in clock ticks.
 * Field 2 is the executable name in parentheses and may itself contain
 * spaces, so the fields are counted from the closing parenthesis rather than
 * by splitting the whole line.
 */
double ReadProcessCpuSeconds() {
  std::ifstream stat("/proc/self/stat");
  if (!stat.good()) return HostMetrics::kUnavailable;

  std::string line;
  std::getline(stat, line);
  const std::string::size_type close = line.rfind(')');
  if (close == std::string::npos) return HostMetrics::kUnavailable;

  // After ") " comes field 3; utime is field 14, so skip 11 more.
  std::string rest = line.substr(close + 1);
  std::size_t pos = 0;
  auto next_token = [&rest, &pos]() {
    while (pos < rest.size() && rest[pos] == ' ') ++pos;
    const std::size_t start = pos;
    while (pos < rest.size() && rest[pos] != ' ') ++pos;
    return rest.substr(start, pos - start);
  };

  std::string token;
  for (int i = 0; i < 12; ++i) {  // fields 3..13 discarded, 14th kept
    token = next_token();
    if (token.empty()) return HostMetrics::kUnavailable;
  }
  const std::string utime = token;
  const std::string stime = next_token();
  if (stime.empty()) return HostMetrics::kUnavailable;

  const auto ticks_per_second = static_cast<int64_t>(sysconf(_SC_CLK_TCK));
  if (ticks_per_second <= 0) return HostMetrics::kUnavailable;

  const double ticks = std::stod(utime) + std::stod(stime);
  return ticks / static_cast<double>(ticks_per_second);
}

/** @brief Resident set size in MiB, from /proc/self/statm. */
double ReadResidentMib() {
  std::ifstream statm("/proc/self/statm");
  if (!statm.good()) return HostMetrics::kUnavailable;

  int64_t total_pages = 0;
  int64_t resident_pages = 0;
  statm >> total_pages >> resident_pages;
  if (!statm.good() && resident_pages == 0) return HostMetrics::kUnavailable;

  const auto page_size = static_cast<int64_t>(sysconf(_SC_PAGESIZE));
  if (page_size <= 0) return HostMetrics::kUnavailable;

  return static_cast<double>(resident_pages) * static_cast<double>(page_size) /
         (1024.0 * 1024.0);
}

/**
 * @brief SoC temperature in Celsius.
 *
 * thermal_zone0 is the CPU zone on a Raspberry Pi. Other boards may expose a
 * different zone or none at all, hence the unavailable path rather than a
 * hard requirement.
 */
double ReadTemperatureCelsius() {
  std::ifstream zone("/sys/class/thermal/thermal_zone0/temp");
  if (!zone.good()) return HostMetrics::kUnavailable;

  double millidegrees = 0.0;
  zone >> millidegrees;
  if (!zone.good() && millidegrees == 0.0) return HostMetrics::kUnavailable;

  return millidegrees / 1000.0;
}

#endif  // __linux__

}  // namespace

bool HostMetrics::IsSupported() const {
#if defined(__linux__)
  return true;
#else
  return false;
#endif
}

int HostMetrics::CoreCount() const {
#if defined(__linux__)
  const auto cores = static_cast<int64_t>(sysconf(_SC_NPROCESSORS_ONLN));
  return cores > 0 ? static_cast<int>(cores) : 1;
#else
  return 1;
#endif
}

double CpuPercentBetween(const HostSample& previous, const HostSample& current,
                         int cores) {
  if (previous.cpu_seconds < 0.0 || current.cpu_seconds < 0.0 ||
      previous.wall_seconds < 0.0 || current.wall_seconds < 0.0) {
    return HostMetrics::kUnavailable;
  }

  const double elapsed = current.wall_seconds - previous.wall_seconds;
  if (elapsed <= 0.0) return HostMetrics::kUnavailable;

  const double used = current.cpu_seconds - previous.cpu_seconds;
  if (used < 0.0) return HostMetrics::kUnavailable;

  const double capacity = elapsed * static_cast<double>(std::max(1, cores));
  const double percent = (used / capacity) * 100.0;
  // Bound scheduling jitter to the displayed capacity range.
  return std::max(0.0, std::min(100.0, percent));
}

HostSample HostMetrics::Sample(const HostSample& previous) const {
  HostSample now;
  now.wall_seconds = NowSeconds();
#if defined(__linux__)
  now.cpu_seconds = ReadProcessCpuSeconds();
  now.rss_mb = ReadResidentMib();
  now.temp_c = ReadTemperatureCelsius();
  now.cpu_percent = CpuPercentBetween(previous, now, CoreCount());
#else
  (void) previous;
#endif
  return now;
}

}  // namespace app
}  // namespace mep_cctv
