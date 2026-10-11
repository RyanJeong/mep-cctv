#!/usr/bin/env bash
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# ============================================================================
# benchmark.sh
# ============================================================================
# Usage:
#   ./benchmark.sh [OPTIONS] [-- <command> [args...]]
#
# Description:
#   Runs a workload while live-sampling embedded-relevant system resources
#   (CPU%, RSS memory, temperature, CPU frequency, throttle status) at a
#   fixed interval. Prints a live table and writes a CSV + Markdown report.
#   Designed for Raspberry Pi 4 but degrades gracefully on macOS.
#
# Options:
#   -i, --interval <sec>   Sampling interval in seconds (default: 1)
#   -d, --duration <sec>   Max duration; kill workload after this time
#   -l, --label <name>     Label for the report files (default: run)
#   -o, --output <dir>     Report output directory (default: ./reports)
#       --no-report        Print live output only; do not write files
#   -v, --verbose          Enable verbose output
#   -h, --help             Show this help message
#
# Embedded notes:
#   Temperature thresholds: WARN at 70 C, HOT at 80 C (Pi 4 throttles).
#   Throttle status via vcgencmd get_throttled reveals under-voltage and
#   thermal throttle events -- critical for embedded reliability assessment.
#
# Examples:
#   ./benchmark.sh -i 1 -d 5 --label smoke -- sleep 5
#   ./benchmark.sh -i 2 --label inference -- python3 run.py
#   ./benchmark.sh --no-report -- stress --cpu 4 --timeout 10
# ============================================================================

set -euo pipefail

color_red()    { printf '\033[31m'; }
color_green()  { printf '\033[32m'; }
color_yellow() { printf '\033[33m'; }
color_blue()   { printf '\033[34m'; }
color_cyan()   { printf '\033[36m'; }
color_bold()   { printf '\033[1m'; }
color_reset()  { printf '\033[0m'; }

info()  { printf '%s\n' "$(color_green)[INFO]$(color_reset) ${1:-}"; }
warn()  { printf '%s\n' "$(color_yellow)[WARN]$(color_reset) ${1:-}"; }
error() { printf '%s\n' "$(color_red)[ERROR]$(color_reset) ${1:-}" >&2; }
debug() { [ "${VERBOSE}" = "1" ] && printf '%s\n' "$(color_cyan)[DEBUG]$(color_reset) ${1:-}" || true; }

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Temperature thresholds (Celsius) -- Pi 4 begins throttling at ~80 C
readonly TEMP_WARN=70
readonly TEMP_HOT=80

show_help() {
  awk '
    /^# =====/ {
      delimiter_count++
      if (delimiter_count == 3) exit
      next
    }
    delimiter_count == 2 && /^# / {
      sub(/^# /, "")
      print
    }
  ' "$0"
}

error_exit() {
  error "$1"
  exit "${2:-1}"
}

# ---------------------------------------------------------------------------
# Resource sampling helpers
# ---------------------------------------------------------------------------

# Returns CPU% for a given PID; empty string if PID is gone.
sample_cpu() {
  local pid="$1"
  local val
  val=$(ps -p "$pid" -o %cpu= 2>/dev/null || true)
  printf '%s' "${val# }"
}

# Returns RSS in MB for a given PID; empty string if PID is gone.
sample_rss_mb() {
  local pid="$1"
  local rss_kb val
  rss_kb=$(ps -p "$pid" -o rss= 2>/dev/null || true)
  rss_kb="${rss_kb# }"
  if [ -n "$rss_kb" ] && [ "$rss_kb" -gt 0 ] 2>/dev/null; then
    val=$(awk "BEGIN{printf \"%.1f\", ${rss_kb}/1024}")
    printf '%s' "$val"
  else
    printf '%s' "${rss_kb:-0}"
  fi
}

# Returns CPU temperature in Celsius.
# Priority: thermal_zone0 (Linux/Pi) -> vcgencmd -> n/a
sample_temp() {
  local raw val
  if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
    raw=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || true)
    if [ -n "$raw" ]; then
      val=$(awk "BEGIN{printf \"%.1f\", ${raw}/1000}")
      printf '%s' "$val"
      return
    fi
  fi
  if command -v vcgencmd >/dev/null 2>&1; then
    raw=$(vcgencmd measure_temp 2>/dev/null || true)
    # output: temp=42.8'C
    val=$(printf '%s' "$raw" | sed "s/temp=//;s/'C//")
    if [ -n "$val" ]; then
      printf '%s' "$val"
      return
    fi
  fi
  printf 'n/a'
}

# Returns current CPU0 frequency in MHz.
# Priority: scaling_cur_freq (Linux/Pi) -> n/a
sample_freq_mhz() {
  local raw val
  if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]; then
    raw=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true)
    if [ -n "$raw" ]; then
      val=$(awk "BEGIN{printf \"%d\", ${raw}/1000}")
      printf '%s' "$val"
      return
    fi
  fi
  printf 'n/a'
}

# Returns throttle status hex string from vcgencmd, or n/a.
# A non-zero value indicates under-voltage or thermal throttling.
sample_throttle() {
  local raw val
  if command -v vcgencmd >/dev/null 2>&1; then
    raw=$(vcgencmd get_throttled 2>/dev/null || true)
    # output: throttled=0x0
    val=$(printf '%s' "$raw" | sed 's/throttled=//')
    if [ -n "$val" ]; then
      printf '%s' "$val"
      return
    fi
  fi
  printf 'n/a'
}

# ---------------------------------------------------------------------------
# System header (captured once at start)
# ---------------------------------------------------------------------------

collect_system_header() {
  local hostname_val arch_val cpu_count_val mem_total_val date_val
  hostname_val=$(hostname 2>/dev/null || printf 'unknown')
  arch_val=$(uname -m 2>/dev/null || printf 'unknown')
  cpu_count_val=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || printf '?')
  if [ -r /proc/meminfo ]; then
    mem_total_val=$(awk '/MemTotal/{printf "%.0f MB", $2/1024}' /proc/meminfo)
  else
    mem_total_val=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f MB", $1/1024/1024}' || printf 'n/a')
  fi
  date_val=$(date '+%Y-%m-%d %H:%M:%S')

  printf '%s|%s|%s|%s|%s' \
    "$hostname_val" "$arch_val" "$cpu_count_val" "$mem_total_val" "$date_val"
}

# ---------------------------------------------------------------------------
# Live table printing
# ---------------------------------------------------------------------------

print_table_header() {
  printf '%s' "$(color_bold)$(color_blue)"
  printf '%-10s  %-8s  %-8s  %-9s  %-8s  %-8s  %-12s\n' \
    'TIMESTAMP' 'ELAPSED' 'CPU%' 'RSS_MB' 'TEMP_C' 'FREQ_MHZ' 'THROTTLE'
  printf '%s' "$(color_reset)"
  printf -- '%.0s-' {1..75}
  printf '\n'
}

# Print a single sample row with temperature-based color.
print_sample_row() {
  local ts="$1" elapsed="$2" cpu="$3" rss="$4" temp="$5" freq="$6" throttle="$7"
  local temp_color
  temp_color=''
  if [ "$temp" != 'n/a' ]; then
    local temp_int
    temp_int=$(printf '%.0f' "$temp" 2>/dev/null || printf '0')
    if [ "$temp_int" -ge "$TEMP_HOT" ] 2>/dev/null; then
      temp_color="$(color_red)"
    elif [ "$temp_int" -ge "$TEMP_WARN" ] 2>/dev/null; then
      temp_color="$(color_yellow)"
    fi
  fi
  printf '%-10s  %-8s  %-8s  %-9s  %s%-8s%s  %-8s  %-12s\n' \
    "$ts" "${elapsed}s" "${cpu}%" "${rss}" \
    "$temp_color" "$temp" "$(color_reset)" \
    "$freq" "$throttle"
}

# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

write_csv_header() {
  local csv_file="$1"
  printf '%s\n' \
    'timestamp,elapsed_s,cpu_pct,rss_mb,temp_c,freq_mhz,throttle' \
    > "$csv_file"
}

append_csv_row() {
  local csv_file="$1"
  local ts="$2" elapsed="$3" cpu="$4" rss="$5" temp="$6" freq="$7" throttle="$8"
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$ts" "$elapsed" "$cpu" "$rss" "$temp" "$freq" "$throttle" \
    >> "$csv_file"
}

write_markdown_report() {
  local md_file="$1" label="$2" sys_header="$3" csv_file="$4" elapsed_total="$5"
  local throttle_seen="$6"

  IFS='|' read -r h_host h_arch h_cpus h_mem h_date <<< "$sys_header"

  # Compute summary stats from CSV (skip header row)
  local samples cpu_avg cpu_peak rss_avg rss_peak temp_max
  samples=$(awk 'NR>1{c++}END{print c+0}' "$csv_file")

  cpu_avg=$(awk -F',' 'NR>1 && $3!=""{s+=$3;c++}END{if(c>0)printf "%.1f",s/c;else print "n/a"}' "$csv_file")
  cpu_peak=$(awk -F',' 'BEGIN{m=-1;found=0} NR>1 && $3!=""{if(!found||$3>m){m=$3;found=1}} END{if(found)printf "%.1f",m;else print "n/a"}' "$csv_file")
  rss_avg=$(awk -F',' 'NR>1 && $4!=""{s+=$4;c++}END{if(c>0)printf "%.1f",s/c;else print "n/a"}' "$csv_file")
  rss_peak=$(awk -F',' 'BEGIN{m=-1;found=0} NR>1 && $4!=""{if(!found||$4>m){m=$4;found=1}} END{if(found)printf "%.1f",m;else print "n/a"}' "$csv_file")
  temp_max=$(awk -F',' 'BEGIN{m=-1;found=0} NR>1 && $5!="n/a" && $5!=""{if(!found||$5>m){m=$5;found=1}} END{if(found)printf "%.1f",m;else print "n/a"}' "$csv_file")

  {
    printf '# Benchmark Report: %s\n\n' "$label"
    printf '## System\n\n'
    printf '| Field        | Value          |\n'
    printf '|:-------------|:---------------|\n'
    printf '| Hostname     | %s |\n' "$h_host"
    printf '| Architecture | %s |\n' "$h_arch"
    printf '| CPU count    | %s |\n' "$h_cpus"
    printf '| Total memory | %s |\n' "$h_mem"
    printf '| Date         | %s |\n' "$h_date"
    printf '\n'
    printf '## Summary\n\n'
    printf '| Metric            | Value      |\n'
    printf '|:------------------|:-----------|\n'
    printf '| Samples           | %s |\n'  "$samples"
    printf '| Duration (s)      | %s |\n'  "$elapsed_total"
    printf '| CPU%% avg          | %s |\n'  "$cpu_avg"
    printf '| CPU%% peak         | %s |\n'  "$cpu_peak"
    printf '| RSS MB avg        | %s |\n'  "$rss_avg"
    printf '| RSS MB peak       | %s |\n'  "$rss_peak"
    printf '| Temp C max        | %s |\n'  "$temp_max"
    printf '| Throttle seen     | %s |\n'  "$throttle_seen"
    printf '\n'
    printf '_CSV data: %s_\n' "$(basename "$csv_file")"
  } > "$md_file"
}

# ---------------------------------------------------------------------------
# Workload selection
# ---------------------------------------------------------------------------

# Resolve the default workload when the user provides no -- command.
# If benchmark.py exists, run it; otherwise run a short self-test.
build_default_workload() {
  local label="$1"
  local py_driver="${SCRIPT_DIR}/benchmark.py"
  if [ -f "$py_driver" ]; then
    debug "Using benchmark.py as default workload"
    WORKLOAD_CMD=("python3" "$py_driver" "--label" "$label")
  else
    warn "benchmark.py not found -- using self-test workload (sleep 5)"
    WORKLOAD_CMD=("sleep" "5")
  fi
}

# ---------------------------------------------------------------------------
# Cleanup trap
# ---------------------------------------------------------------------------

WORKLOAD_PID=''
SAMPLER_TMPFILE=''

cleanup() {
  if [ -n "$WORKLOAD_PID" ] && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    debug "cleanup: terminating workload PID ${WORKLOAD_PID}"
    # Try process-group kill (Linux/setsid); fall back to single-PID kill.
    if command -v setsid >/dev/null 2>&1; then
      kill -- "-${WORKLOAD_PID}" 2>/dev/null || true
    fi
    kill "$WORKLOAD_PID" 2>/dev/null || true
  fi
  if [ -n "$SAMPLER_TMPFILE" ] && [ -f "$SAMPLER_TMPFILE" ]; then
    rm -f -- "$SAMPLER_TMPFILE"
  fi
}

# ---------------------------------------------------------------------------
# Main sampling loop
# ---------------------------------------------------------------------------

run_benchmark() {
  local interval="$1"
  local duration="$2"
  local label="$3"
  local output_dir="$4"
  local no_report="$5"
  local sys_header
  sys_header=$(collect_system_header)

  # Resolve output paths
  local ts_stamp csv_file md_file
  ts_stamp=$(date '+%Y%m%d-%H%M%S')
  csv_file="${output_dir}/${label}-${ts_stamp}.csv"
  md_file="${output_dir}/${label}-${ts_stamp}.md"

  if [ "$no_report" = "0" ]; then
    mkdir -p -- "$output_dir"
    write_csv_header "$csv_file"
    debug "CSV: ${csv_file}"
    debug "MD:  ${md_file}"
  fi

  # Print system header summary
  IFS='|' read -r h_host h_arch h_cpus h_mem h_date <<< "$sys_header"
  printf '\n'
  printf '%s' "$(color_bold)"
  printf 'Benchmark: %s  |  host: %s  |  arch: %s  |  cpus: %s  |  mem: %s\n' \
    "$label" "$h_host" "$h_arch" "$h_cpus" "$h_mem"
  printf '%s\n' "$(color_reset)"

  # Launch workload in its own process group.
  # setsid is Linux-only; fall back to plain background exec on macOS.
  debug "Launching: ${WORKLOAD_CMD[*]}"
  if command -v setsid >/dev/null 2>&1; then
    setsid "${WORKLOAD_CMD[@]}" &
  else
    "${WORKLOAD_CMD[@]}" &
  fi
  WORKLOAD_PID=$!
  debug "Workload PID: ${WORKLOAD_PID}"

  print_table_header

  local start_epoch elapsed sample_ts cpu rss temp freq throttle
  local throttle_seen='0x0'
  local duration_cap="${duration}"
  start_epoch=$(date +%s)

  while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
    elapsed=$(( $(date +%s) - start_epoch ))

    # Duration cap: kill workload if exceeded
    if [ -n "$duration_cap" ] && [ "$elapsed" -ge "$duration_cap" ]; then
      debug "Duration cap ${duration_cap}s reached; stopping workload"
      if command -v setsid >/dev/null 2>&1; then
        kill -- "-${WORKLOAD_PID}" 2>/dev/null || true
      fi
      kill "$WORKLOAD_PID" 2>/dev/null || true
      break
    fi

    sample_ts=$(date '+%H:%M:%S')
    cpu=$(sample_cpu "$WORKLOAD_PID")
    rss=$(sample_rss_mb "$WORKLOAD_PID")
    temp=$(sample_temp)
    freq=$(sample_freq_mhz)
    throttle=$(sample_throttle)

    # Track if any non-zero throttle was observed
    if [ "$throttle" != 'n/a' ] && [ "$throttle" != '0x0' ] && [ "$throttle" != '0x00000' ]; then
      throttle_seen="$throttle"
      warn "Throttle event detected: ${throttle}"
    fi

    # Default cpu/rss to 0 if workload just exited mid-sample
    cpu="${cpu:-0}"
    rss="${rss:-0}"

    print_sample_row "$sample_ts" "$elapsed" "$cpu" "$rss" "$temp" "$freq" "$throttle"

    if [ "$no_report" = "0" ]; then
      append_csv_row "$csv_file" "$sample_ts" "$elapsed" "$cpu" "$rss" "$temp" "$freq" "$throttle"
    fi

    sleep "$interval" &
    local sleep_pid=$!
    # Wake immediately if workload dies while sleeping
    wait "$sleep_pid" 2>/dev/null || true
  done

  # Wait for workload to finish; capture exit code
  local workload_exit=0
  wait "$WORKLOAD_PID" 2>/dev/null || workload_exit=$?
  WORKLOAD_PID=''

  local total_elapsed
  total_elapsed=$(( $(date +%s) - start_epoch ))
  printf -- '%.0s-' {1..75}
  printf '\n'
  info "Workload finished in ${total_elapsed}s (exit ${workload_exit})"

  if [ "$no_report" = "0" ]; then
    write_markdown_report "$md_file" "$label" "$sys_header" "$csv_file" "$total_elapsed" "$throttle_seen"
    info "CSV report:      ${csv_file}"
    info "Markdown report: ${md_file}"
  fi
}

# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------

WORKLOAD_CMD=()

main() {
  local interval=1
  local duration=''
  local label='run'
  local output_dir="${SCRIPT_DIR}/reports"
  local no_report=0
  VERBOSE=0

  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        show_help
        exit 0
        ;;
      -v|--verbose)
        VERBOSE=1
        shift
        ;;
      -i|--interval)
        [ "$#" -ge 2 ] || error_exit "--interval requires an argument"
        interval="$2"
        shift 2
        ;;
      -d|--duration)
        [ "$#" -ge 2 ] || error_exit "--duration requires an argument"
        duration="$2"
        shift 2
        ;;
      -l|--label)
        [ "$#" -ge 2 ] || error_exit "--label requires an argument"
        label="$2"
        shift 2
        ;;
      -o|--output)
        [ "$#" -ge 2 ] || error_exit "--output requires an argument"
        output_dir="$2"
        shift 2
        ;;
      --no-report)
        no_report=1
        shift
        ;;
      --)
        shift
        WORKLOAD_CMD=("$@")
        break
        ;;
      -*)
        error_exit "Unknown option: $1"
        ;;
      *)
        error_exit "Unexpected argument: $1 (use -- to separate the workload command)"
        ;;
    esac
  done

  if [ "${#WORKLOAD_CMD[@]}" -eq 0 ]; then
    build_default_workload "$label"
  fi

  trap cleanup EXIT INT TERM

  SAMPLER_TMPFILE=$(mktemp)

  run_benchmark "$interval" "$duration" "$label" "$output_dir" "$no_report"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
