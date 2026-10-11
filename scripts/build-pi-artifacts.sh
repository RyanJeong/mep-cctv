#!/usr/bin/env bash

# ============================================================================
# scripts/build-pi-artifacts.sh -- build the Raspberry Pi 4 artifacts
# ============================================================================
# Usage:
#   ./scripts/build-pi-artifacts.sh [OPTIONS]
#
# Description:
#   Cross-builds the aarch64 artifact for each mode into dist/pi4-<mode>/,
#   so they can be copied to a Pi and compared side by side.
#
#     onnx    the original model, run by ONNX Runtime (bundles the runtime
#             and the .onnx file)
#     native  the same model compiled to machine code by TVM and fully
#             optimized; a single shared library with nothing beside it
#
#   The native mode links the REAL TVM-compiled model, so the generated
#   sources must exist first. Apply the committed tuning database as described
#   in models/README.md, or use --placeholder for a development-only build.
#
#   Each folder is self-contained: the binaries carry an $ORIGIN runpath, so
#   no LD_LIBRARY_PATH is needed on the Pi.
#
# Options:
#   -m, --mode <name>   Build only one mode (onnx | native)
#   -p, --placeholder   native links the stand-in model instead of the real
#                       one: builds without the TVM toolchain, runs the whole
#                       pipeline, and reports no detections
#   -o, --out <dir>     Output directory (default: dist)
#   -h, --help          Show help
# ============================================================================

set -euo pipefail

color_red() { printf '\033[31m'; }
color_green() { printf '\033[32m'; }
color_yellow() { printf '\033[33m'; }
color_cyan() { printf '\033[36m'; }
color_bold() { printf '\033[1m'; }
color_reset() { printf '\033[0m'; }

info() { printf '%s\n' "$(color_green)[INFO]$(color_reset) ${1:-}"; }
warn() { printf '%s\n' "$(color_yellow)[WARN]$(color_reset) ${1:-}"; }
error() { printf '%s\n' "$(color_red)[ERROR]$(color_reset) ${1:-}" >&2; }

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly DOCKERFILE="${REPO_ROOT}/docker/Dockerfile.crossbuild"

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

# Accept the two public mode names.
mode_is_valid() {
	case "$1" in
	onnx | native) return 0 ;;
	*) return 1 ;;
	esac
}

build_one() {
	local mode="$1"
	local out_dir="$2"
	local use_generated="$3"
	local dest model

	dest="${out_dir}/pi4-${mode}"
	# Real artifacts require generated model files rather than automatic fallback.
	model="real"
	if [ "${use_generated}" -eq 0 ]; then
		model="placeholder"
	fi
	# MODEL selects native input only; onnx loads its model file at runtime.
	if [ "${mode}" = "native" ]; then
		info "$(color_bold)${mode}$(color_reset) (MODE=${mode} MODEL=${model}) -> ${dest}"
	else
		info "$(color_bold)${mode}$(color_reset) (MODE=${mode}) -> ${dest}"
	fi
	rm -rf "${dest}"

	declare -a cmd=(
		docker buildx build
		"--file=${DOCKERFILE}"
		"--target=artifact"
		"--build-arg=TARGET_ARCH=aarch64"
		"--build-arg=MODE=${mode}"
		"--output=type=local,dest=${dest}"
		"${REPO_ROOT}"
	)
	if [ "${mode}" = "native" ]; then
		cmd+=("--build-arg=MODEL=${model}")
	fi
	"${cmd[@]}"

	# Copy the ONNX model after exporting the Docker artifact.
	if [ "${mode}" = "onnx" ]; then
		if [ -f "${REPO_ROOT}/models/onnx/yolov8n.onnx" ]; then
			cp "${REPO_ROOT}/models/onnx/yolov8n.onnx" "${dest}/"
			info "  bundled yolov8n.onnx"
		else
			warn "  models/onnx/yolov8n.onnx not found -- the onnx mode will not"
			warn "  start without it. Export it first, then re-run."
		fi
	fi

	info "  $(color_green)done$(color_reset): $(du -sh "${dest}" | cut -f1)"
}

main() {
	local only_mode=""
	local out_dir="${REPO_ROOT}/dist"
	local use_generated=1

	while [ $# -gt 0 ]; do
		case "$1" in
		-m | --mode)
			shift
			only_mode="${1:?--mode requires onnx or native}"
			;;
		-o | --out)
			shift
			out_dir="${1:?--out requires a directory}"
			;;
		-p | --placeholder)
			use_generated=0
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

	if ! command -v docker >/dev/null 2>&1; then
		error "docker not found. Install Docker Desktop or Docker Engine."
		exit 1
	fi

	# Generated kernels and weights are both needed for a real native artifact.
	local generated_dir="${REPO_ROOT}/src/backend_native/generated"
	if [ "${use_generated}" -eq 1 ]; then
		local missing=""
		local gen_file
		for gen_file in yolov8n_tuned_kernels.o yolov8n_tuned_blob.cc; do
			[ -f "${generated_dir}/${gen_file}" ] || missing="${missing} ${gen_file}"
		done
		if [ -n "${missing}" ]; then
			error "TVM-generated sources are missing:${missing}"
			error ""
			error "Apply the committed tuning database (about two minutes):"
			error "  .venv-convert/bin/python models/convert/tvm_compile.py \\"
			error "    --onnx models/onnx/yolov8n.onnx \\"
			error "    --input-name images --input-shape 1,3,320,320 \\"
			error "    --out yolov8n_tuned --target arm64 \\"
			error "    --emit-embed src/backend_native/generated \\"
			error "    --tune-db models/tuning/rpi4"
			error ""
			error "After clean.sh --deep the inputs above are gone too:"
			error "  python3.11 -m venv .venv-convert &&"
			error "    .venv-convert/bin/pip install -r models/convert/requirements.txt"
			error "  ./models/download.sh"
			error "Re-tuning from scratch (hours) is only for different hardware:"
			error "  ./scripts/tune-on-pi.sh --ssh <user>@<pi>"
			error "Or pass --placeholder to build without a real model."
			exit 1
		fi
	fi

	mkdir -p "${out_dir}"

	declare -a modes=("onnx" "native")
	if [ -n "${only_mode}" ]; then
		case "${only_mode}" in
		onnx | native) modes=("${only_mode}") ;;
		*)
			error "Invalid mode '${only_mode}'. Use onnx or native."
			exit 1
			;;
		esac
		mode_is_valid "${only_mode}" || exit 1
	fi

	local mode
	for mode in "${modes[@]}"; do
		build_one "${mode}" "${out_dir}" "${use_generated}"
	done

	printf '\n'
	info "$(color_bold)Artifacts ready$(color_reset)"
	for mode in "${modes[@]}"; do
		printf '  %s\n' "$(color_cyan)${out_dir}/pi4-${mode}$(color_reset)"
	done
	printf '\n'
	info "Copy them to the Pi with:"
	printf '  scp -r %s/pi4-* pi@raspberrypi.local:~/\n' "${out_dir}"
	info "Then see the \"Run on a Raspberry Pi 4\" section of README.md"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
	main "$@"
fi
