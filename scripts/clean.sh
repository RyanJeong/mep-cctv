#!/usr/bin/env bash
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# ============================================================================
# scripts/clean.sh  --  mep-cctv build-artifact cleaner
# ============================================================================
# Usage:
#   ./scripts/clean.sh [OPTIONS]
#
# Description:
#   Removes build artifacts and generated files under the repository root.
#   All deletions are confined to paths under REPO_ROOT -- never system paths.
#
# Options:
#   -h, --help       Show this help message
#   -v, --verbose    Print each candidate as it is evaluated
#   -n, --dry-run    Print what WOULD be removed; delete nothing
#       --deep       Also remove virtualenvs and downloaded model files
#   -y, --yes        Skip the confirmation prompt
#
# Examples:
#   ./scripts/clean.sh
#   ./scripts/clean.sh --dry-run
#   ./scripts/clean.sh --deep --yes
#   ./scripts/clean.sh --verbose --dry-run
# ============================================================================

set -euo pipefail

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
debug() { printf '%s\n' "$(color_cyan)[DEBUG]$(color_reset) ${1:-}"; }

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

error_exit() {
	error "$1"
	exit "${2:-1}"
}

# ---------------------------------------------------------------------------
# Safety guard: confirm path is non-empty and is REPO_ROOT or descends from it.
# Used for search roots (find base directories).
# Returns 1 (and emits an error) if the guard fails.
# ---------------------------------------------------------------------------
_assert_rooted_in_repo() {
	local path="$1"
	[[ -n "${path}" ]] || return 1
	local canonical="${path%/}"
	case "${canonical}" in
	"${REPO_ROOT}" | "${REPO_ROOT}/"*)
		return 0
		;;
	*)
		error "Safety guard: search root is outside repo: ${canonical}"
		return 1
		;;
	esac
}

# ---------------------------------------------------------------------------
# Safety guard: confirm path is non-empty and STRICTLY descends from REPO_ROOT.
# Used for deletion targets -- never allow removing REPO_ROOT itself.
# Returns 1 (and emits an error) if the guard fails.
# ---------------------------------------------------------------------------
_assert_under_repo() {
	local path="$1"
	[[ -n "${path}" ]] || return 1
	local canonical="${path%/}"
	case "${canonical}" in
	"${REPO_ROOT}/"*)
		return 0
		;;
	*)
		error "Safety guard: refusing to touch path outside repo: ${canonical}"
		return 1
		;;
	esac
}

# ---------------------------------------------------------------------------
# Append a path to the global TARGETS array after safety + existence checks.
# Usage: _enqueue <path> <verbose:0|1>
# ---------------------------------------------------------------------------
_enqueue() {
	local path="$1"
	local verbose="$2"

	_assert_under_repo "${path}" || return 0

	if [[ ! -e "${path}" && ! -L "${path}" ]]; then
		[[ "${verbose}" -eq 1 ]] && debug "skip (not found): ${path#"${REPO_ROOT}/"}"
		return 0
	fi

	TARGETS+=("${path}")
	info "found: ${path#"${REPO_ROOT}/"}"
}

# ---------------------------------------------------------------------------
# Find and enqueue all directories matching a name pattern under a root.
# Uses find -print0 / read -d '' for safe iteration over any filename.
# ---------------------------------------------------------------------------
_enqueue_named_dirs() {
	local search_root="$1"
	local name_pattern="$2"
	local verbose="$3"

	_assert_rooted_in_repo "${search_root}" || return 0
	[[ -d "${search_root}" ]] || return 0

	local path
	while IFS= read -r -d '' path; do
		_enqueue "${path}" "${verbose}"
	done < <(find "${search_root}" -type d -name "${name_pattern}" -prune -print0 2>/dev/null)
}

# ---------------------------------------------------------------------------
# Find and enqueue all files matching a name pattern under a root.
# ---------------------------------------------------------------------------
_enqueue_named_files() {
	local search_root="$1"
	local name_pattern="$2"
	local verbose="$3"

	_assert_rooted_in_repo "${search_root}" || return 0
	[[ -d "${search_root}" ]] || return 0

	local path
	while IFS= read -r -d '' path; do
		_enqueue "${path}" "${verbose}"
	done < <(find "${search_root}" -type f -name "${name_pattern}" -print0 2>/dev/null)
}

# ---------------------------------------------------------------------------
# Category collectors -- each appends to the global TARGETS array.
# ---------------------------------------------------------------------------

_collect_build_dirs() {
	local verbose="$1"
	local dir
	for dir in \
		"${REPO_ROOT}/build" \
		"${REPO_ROOT}/build-"*; do
		[[ -e "${dir}" || -L "${dir}" ]] || continue
		_enqueue "${dir}" "${verbose}"
	done
}

_collect_cmake_stragglers() {
	local verbose="$1"
	local item
	for item in \
		"${REPO_ROOT}/CMakeCache.txt" \
		"${REPO_ROOT}/CMakeFiles" \
		"${REPO_ROOT}/cmake_install.cmake" \
		"${REPO_ROOT}/_deps"; do
		_enqueue "${item}" "${verbose}"
	done
}

_collect_coverage() {
	local verbose="$1"
	local pattern
	for pattern in "*.gcda" "*.gcno" "coverage.info" "*.profraw" "*.profdata"; do
		_enqueue_named_files "${REPO_ROOT}" "${pattern}" "${verbose}"
	done
	local item
	for item in \
		"${REPO_ROOT}/.coverage" \
		"${REPO_ROOT}/coverage"; do
		_enqueue "${item}" "${verbose}"
	done
}

_collect_python_caches() {
	local verbose="$1"

	# Find __pycache__ dirs, excluding virtualenvs and build dirs to avoid
	# enumerating thousands of cache entries that will be removed with their
	# parent venv/build directory anyway.
	local path
	while IFS= read -r -d '' path; do
		_enqueue "${path}" "${verbose}"
	done < <(find "${REPO_ROOT}" \
		\( -name ".venv" -o -name ".venv-*" -o -name "build" -o -name "build-*" \) -prune \
		-o -type d -name "__pycache__" -prune -print0 2>/dev/null)

	while IFS= read -r -d '' path; do
		_enqueue "${path}" "${verbose}"
	done < <(find "${REPO_ROOT}" \
		\( -name ".venv" -o -name ".venv-*" -o -name "build" -o -name "build-*" \) -prune \
		-o -type f -name "*.pyc" -print0 2>/dev/null)

	local item
	for item in \
		"${REPO_ROOT}/.pytest_cache" \
		"${REPO_ROOT}/.mypy_cache"; do
		_enqueue "${item}" "${verbose}"
	done
}

_collect_bench_reports() {
	local verbose="$1"
	local reports_dir="${REPO_ROOT}/bench/reports"
	[[ -d "${reports_dir}" ]] || return 0
	_assert_under_repo "${reports_dir}" || return 0

	local path
	while IFS= read -r -d '' path; do
		[[ "${path##*/}" == ".gitkeep" ]] && continue
		_enqueue "${path}" "${verbose}"
	done < <(find "${reports_dir}" -maxdepth 1 -not -name ".gitkeep" \
		\( -type f -o -type l \) -print0 2>/dev/null)
}

_collect_dist() {
	local verbose="$1"
	_enqueue "${REPO_ROOT}/dist" "${verbose}"
}

_collect_tvm_generated() {
	local verbose="$1"
	local gen_dir="${REPO_ROOT}/src/backend_native/generated"
	[[ -d "${gen_dir}" ]] || return 0
	_assert_under_repo "${gen_dir}" || return 0

	# Remove current and legacy generated outputs, not hand-written backend files.
	local path
	for path in \
		"${gen_dir}"/*_kernels.o \
		"${gen_dir}"/*_kernels.c \
		"${gen_dir}"/*_blob.cc \
		"${gen_dir}"/*_register.cc; do
		[[ -e "${path}" ]] || continue
		_enqueue "${path}" "${verbose}"
	done
}

_collect_deep() {
	local verbose="$1"
	warn "Deep clean: will also remove virtualenvs and downloaded model files."
	warn "These must be re-created / re-downloaded before the next build."

	local venv
	for venv in \
		"${REPO_ROOT}/.venv" \
		"${REPO_ROOT}/.venv-"*; do
		[[ -e "${venv}" ]] || continue
		_enqueue "${venv}" "${verbose}"
	done

	local onnx_dir="${REPO_ROOT}/models/onnx"
	if [[ -d "${onnx_dir}" ]]; then
		_assert_under_repo "${onnx_dir}" || return 0
		local model_file
		while IFS= read -r -d '' model_file; do
			_enqueue "${model_file}" "${verbose}"
		done < <(find "${onnx_dir}" \( -name "*.onnx" -o -name "*.tflite" \) \
			-type f -print0 2>/dev/null)
	fi

	_enqueue "${REPO_ROOT}/models/cache" "${verbose}"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
	local dry_run=0
	local verbose=0
	local deep=0
	local yes=0

	while [[ $# -gt 0 ]]; do
		case "$1" in
		-h | --help)
			show_help
			exit 0
			;;
		-v | --verbose) verbose=1 ;;
		-n | --dry-run) dry_run=1 ;;
		--deep) deep=1 ;;
		-y | --yes) yes=1 ;;
		*) error_exit "Unknown option: $1" ;;
		esac
		shift
	done

	info "Repository root: ${REPO_ROOT}"
	[[ "${dry_run}" -eq 1 ]] && warn "Dry-run mode -- no files will be deleted."

	# Global target list populated by the collectors.
	declare -a TARGETS=()

	# Phase 1: discover targets (always a scan pass, no deletion).
	_collect_build_dirs "${verbose}"
	_collect_cmake_stragglers "${verbose}"
	_collect_coverage "${verbose}"
	_collect_python_caches "${verbose}"
	_collect_bench_reports "${verbose}"
	_collect_dist "${verbose}"
	_collect_tvm_generated "${verbose}"
	[[ "${deep}" -eq 1 ]] && _collect_deep "${verbose}"

	local total="${#TARGETS[@]}"

	if [[ "${total}" -eq 0 ]]; then
		info "Nothing to remove -- repository is already clean."
		return 0
	fi

	if [[ "${dry_run}" -eq 1 ]]; then
		info "Dry-run complete: ${total} item(s) would be removed."
		return 0
	fi

	# Phase 2: confirm before deleting (skip when -y/--yes).
	if [[ "${yes}" -eq 0 ]]; then
		printf '%s' "$(color_yellow)[WARN]$(color_reset) About to remove ${total} item(s). Proceed? [y/N] "
		local answer
		read -r answer
		case "${answer}" in
		[Yy] | [Yy][Ee][Ss]) ;;
		*)
			info "Aborted -- no files were removed."
			return 0
			;;
		esac
	fi

	# Phase 3: delete.
	local path
	for path in "${TARGETS[@]}"; do
		_assert_under_repo "${path}" || continue
		[[ -e "${path}" || -L "${path}" ]] || continue
		rm -rf -- "${path}"
		info "removed: ${path#"${REPO_ROOT}/"}"
	done

	info "Done -- removed ${total} item(s)."
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
	main "$@"
fi
