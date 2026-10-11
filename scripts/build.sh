#!/usr/bin/env bash
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# ============================================================================
# scripts/build.sh  --  mep-cctv CMake build wrapper
# ============================================================================
# Usage:
#   ./scripts/build.sh [OPTIONS]
#
# Options:
#   --mode <native|onnx>          Select the build mode (default: native)
#   --scalar-kernels              Build the scalar reference kernels instead
#                                 of SIMD (native only; for testing)

#   --arch <host|arm64|x86_64>    Target arch (host = build natively)
#   --coverage                    Enable code-coverage instrumentation
#   --asan                        Enable AddressSanitizer + UBSan
#   --no-test                     Skip ctest after build
#   --build-type <Debug|Release>  CMake build type (default: Release)
#   -v, --verbose                 Print every cmake/ctest command
#   -h, --help                    Show this help
#
# Examples:
#   ./scripts/build.sh --mode native
#   ./scripts/build.sh --mode onnx --asan --build-type Debug
#   ./scripts/build.sh --arch arm64
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Colour helpers (portable: printf '\033[..m' pattern, ASCII only)
# ----------------------------------------------------------------------------
color_red() { printf '\033[31m'; }
color_green() { printf '\033[32m'; }
color_yellow() { printf '\033[33m'; }
color_blue() { printf '\033[34m'; }
color_cyan() { printf '\033[36m'; }
color_bold() { printf '\033[1m'; }
color_reset() { printf '\033[0m'; }

info() { printf '%s\n' "$(color_green)[INFO]$(color_reset) ${1:-}"; }
warn() { printf '%s\n' "$(color_yellow)[WARN]$(color_reset) ${1:-}"; }
error() { printf '%s\n' "$(color_red)[ERROR]$(color_reset) ${1:-}" >&2; }

# ----------------------------------------------------------------------------
# Locate repository root (one level above scripts/)
# ----------------------------------------------------------------------------
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

# ----------------------------------------------------------------------------
# Cross-arch: delegate to Docker buildx
# ----------------------------------------------------------------------------
run_cross_build() {
	local arch="$1"
	local mode="$2"
	local scalar_kernels="$3"
	local verbose="$4"

	local platform="linux/${arch}"
	info "Cross build: platform=${platform} mode=${mode} scalar_kernels=${scalar_kernels}"

	if ! command -v docker >/dev/null 2>&1; then
		warn "docker not found. Run this command manually:"
		printf '%s\n' "$(color_cyan)  docker buildx build \\$(color_reset)"
		printf '%s\n' "$(color_cyan)    --platform ${platform} \\$(color_reset)"
		printf '%s\n' "$(color_cyan)    --file ${REPO_ROOT}/docker/Dockerfile.crossbuild \\$(color_reset)"
		printf '%s\n' "$(color_cyan)    --target artifact \\$(color_reset)"
		printf '%s\n' "$(color_cyan)    --build-arg MODE=${mode} \\$(color_reset)"
		printf '%s\n' "$(color_cyan)    --build-arg SCALAR_KERNELS=${scalar_kernels} \\$(color_reset)"
		printf '%s\n' "$(color_cyan)    --output type=local,dest=${REPO_ROOT}/dist \\$(color_reset)"
		printf '%s\n' "$(color_cyan)    .$(color_reset)"
		return 0
	fi

	local dist_dir="${REPO_ROOT}/dist/${arch}"
	mkdir -p "${dist_dir}"

	declare -a docker_cmd=(
		docker buildx build
		"--platform=${platform}"
		"--file=${REPO_ROOT}/docker/Dockerfile.crossbuild"
		"--target=artifact"
		"--build-arg=MODE=${mode}"
		"--build-arg=SCALAR_KERNELS=${scalar_kernels}"
		"--output=type=local,dest=${dist_dir}"
	)
	if [[ "${verbose}" -eq 1 ]]; then
		docker_cmd+=(--progress plain)
		info "Running: ${docker_cmd[*]}"
	fi
	docker_cmd+=("${REPO_ROOT}")
	"${docker_cmd[@]}"
	info "Artifact written to ${dist_dir}/"
}

# ----------------------------------------------------------------------------
# Host (native) build
# ----------------------------------------------------------------------------
run_host_build() {
	local mode="$1"
	local scalar_kernels="$2"
	local build_type="$3"
	local coverage="$4"
	local asan="$5"
	local no_test="$6"
	local verbose="$7"

	local build_dir="${REPO_ROOT}/build-script"

	info "$(color_bold)mep-cctv build$(color_reset)"
	info "  mode           = ${mode}"
	info "  scalar kernels = ${scalar_kernels}"
	info "  build type = ${build_type}"
	info "  coverage   = ${coverage}"
	info "  asan       = ${asan}"
	info "  build dir  = ${build_dir}"

	local build_tests="ON"
	if [[ "${no_test}" -eq 1 ]]; then
		build_tests="OFF"
	fi

	declare -a cmake_args=(
		"-DCMAKE_BUILD_TYPE=${build_type}"
		"-DCCTV_MODE=${mode}"
		"-DCCTV_SCALAR_KERNELS=${scalar_kernels}"
		"-DENABLE_COVERAGE=${coverage}"
		"-DENABLE_ASAN=${asan}"
		"-DBUILD_TESTS=${build_tests}"
	)

	if [[ "${verbose}" -eq 1 ]]; then
		info "cmake -B '${build_dir}' ${cmake_args[*]}"
	fi
	cmake -B "${build_dir}" "${cmake_args[@]}"

	declare -a build_cmd=(cmake --build "${build_dir}" --parallel)
	if [[ "${verbose}" -eq 1 ]]; then
		build_cmd+=(--verbose)
		info "Running: ${build_cmd[*]}"
	fi
	"${build_cmd[@]}"
	info "$(color_green)Build succeeded.$(color_reset)"

	if [[ "${no_test}" -eq 0 ]]; then
		info "Running tests..."
		declare -a ctest_args=(ctest --test-dir "${build_dir}" --output-on-failure)
		if [[ "${verbose}" -eq 1 ]]; then
			ctest_args+=(--verbose)
			info "Running: ${ctest_args[*]}"
		fi
		"${ctest_args[@]}"
		info "$(color_green)All tests passed.$(color_reset)"
	fi
}

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
main() {
	local mode="native"
	local scalar_kernels="OFF"
	local arch="host"
	local coverage="OFF"
	local asan="OFF"
	local no_test=0
	local build_type="Release"
	local verbose=0

	while [[ $# -gt 0 ]]; do
		case "$1" in
		--mode)
			shift
			mode="${1:?--mode requires an argument: native or onnx}"
			;;
		--scalar-kernels)
			scalar_kernels="ON"
			;;
		--arch)
			shift
			arch="${1:?--arch requires an argument: host, arm64, or x86_64}"
			;;
		--coverage)
			coverage="ON"
			;;
		--asan)
			asan="ON"
			;;
		--no-test)
			no_test=1
			;;
		--build-type)
			shift
			build_type="${1:?--build-type requires Debug or Release}"
			;;
		-v | --verbose)
			verbose=1
			;;
		-h | --help)
			show_help
			exit 0
			;;
		*)
			error "Unknown option: $1"
			show_help
			exit 1
			;;
		esac
		shift
	done

	# Validate
	case "${mode}" in
	native | onnx) ;;
	*)
		error "Invalid mode '${mode}'. Use native or onnx."
		exit 1
		;;
	esac

	case "${arch}" in
	host | arm64 | x86_64) ;;
	*)
		error "Invalid arch '${arch}'. Use host, arm64, or x86_64."
		exit 1
		;;
	esac

	if [[ "${arch}" != "host" ]]; then
		run_cross_build "${arch}" "${mode}" "${scalar_kernels}" "${verbose}"
	else
		run_host_build \
			"${mode}" "${scalar_kernels}" "${build_type}" \
			"${coverage}" "${asan}" "${no_test}" "${verbose}"
	fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
	main "$@"
fi
