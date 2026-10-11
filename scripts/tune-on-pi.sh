#!/usr/bin/env bash

# ============================================================================
# scripts/tune-on-pi.sh -- autotune the model against the real Raspberry Pi
# ============================================================================
# Usage:
#   ./scripts/tune-on-pi.sh --pi <host> [OPTIONS]
#
# Description:
#   Runs MetaSchedule with candidates BUILT here and MEASURED on the Pi, then
#   emits the tuned sources for the native build.
#
#   Why measure remotely: the best loop tiling depends on cache sizes and core
#   count. A schedule tuned against a laptop is tuned for the wrong computer,
#   so the numbers it picks are wrong for a Cortex-A72 with 1 MB of L2 and
#   four cores. Only measuring on the target fixes that.
#
#   This script starts a tracker locally, tells you how to point the Pi at it,
#   waits for the Pi to register, and then tunes.
#
# One-time setup ON THE PI (skip entirely when using --ssh):
#   sudo apt install -y python3-pip
#   pip3 install --break-system-packages apache-tvm==0.26.* numpy
#
# With --ssh, this script does that setup for you over SSH, starts the
# rpc_server on the Pi, and shuts it down again when tuning ends.
#
# When the Pi's system Python cannot load TVM:
#   Python 3.10-3.13 all work in a clean environment, but a pip
#   --break-system-packages install into the system Python can leave a mix
#   that segfaults inside dlopen on "import tvm" rather than failing cleanly
#   (observed on Debian 13 with its system 3.13). Install a private
#   interpreter on the Pi and point --pi-python at it; nothing system-wide
#   changes, and "rm -rf ~/.local/cctv-tune" undoes it:
#
#     ssh <pi> '
#       mkdir -p ~/.local/cctv-tune && cd ~/.local/cctv-tune
#       curl -fsSL <python-build-standalone 3.11 aarch64 install_only URL> \
#         -o py.tar.gz && tar xzf py.tar.gz && rm py.tar.gz'
#
#   then pass --pi-python ~/.local/cctv-tune/python/bin/python3.11
#
# Options:
#   -p, --pi <host>      Pi hostname or IP (required; used for the reminder)
#   -s, --ssh <target>   user@host to drive the Pi over SSH (e.g. pi@192.0.2.10)
#       --pi-python <p>  Interpreter to use on the Pi (default: python3).
#                        Use when the system Python cannot load TVM --
#                        see the note below.
#   -k, --key <name>     Device key the Pi registers under (default: rpi4)
#   -t, --trials <n>     Tuning trial budget (default: 3000)
#       --port <n>       Tracker port (default: 9190)
#       --onnx <path>    Model to tune (default: models/onnx/yolov8n.onnx)
#       --no-wait        Do not wait for the Pi; assume it is already up
#   -h, --help           Show help
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
readonly VENV_PYTHON="${REPO_ROOT}/.venv-convert/bin/python"
readonly GENERATED_DIR="${REPO_ROOT}/src/backend_native/generated"

TRACKER_PID=""
SSH_TARGET=""
PI_RPC_STARTED=0
PI_PYTHON="python3"

# The wheel index only publishes aarch64 builds, and the schedules being tuned
# are compiled for aarch64 anyway. Tuning against a 32-bit or single-core board
# would silently produce numbers for the wrong machine, so the arch is a gate
# rather than a warning.
readonly PI_REQUIRED_ARCH="aarch64"
readonly PI_TVM_VERSION="0.26.0"

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

cleanup() {
	if [ "${PI_RPC_STARTED}" -eq 1 ]; then
		info "stopping rpc_server on the Pi"
		ssh_run "pkill -f 'tvm[.]exec[.]rpc_server'; true" >/dev/null 2>&1 || true
	fi
	if [ -n "${TRACKER_PID}" ] && kill -0 "${TRACKER_PID}" 2>/dev/null; then
		info "stopping RPC tracker (pid ${TRACKER_PID})"
		kill "${TRACKER_PID}" 2>/dev/null || true
	fi
}

# ---------------------------------------------------------------------------
# Driving the Pi over SSH (--ssh)
# ---------------------------------------------------------------------------

ssh_run() {
	ssh -o BatchMode=yes -o ConnectTimeout=10 "${SSH_TARGET}" "$@"
}

# Refuse anything that is not a 64-bit multi-core Pi. Several boards answer to
# raspberrypi.local on a typical LAN, and an armv6l Pi 1 will accept the SSH
# login just as happily as the Pi 4 -- it simply cannot run the wheel, and any
# timings it produced would describe the wrong computer.
preflight_pi() {
	info "checking ${SSH_TARGET}"
	local facts arch cores model
	if ! facts="$(ssh_run '
		printf "arch=%s\n" "$(uname -m)"
		printf "cores=%s\n" "$(nproc)"
		printf "model=%s\n" "$(tr -d "\0" </proc/device-tree/model 2>/dev/null)"
	' 2>&1)"; then
		error "cannot reach ${SSH_TARGET} over SSH:"
		printf '%s\n' "${facts}" >&2
		return 1
	fi

	arch="$(printf '%s\n' "${facts}" | sed -n 's/^arch=//p')"
	cores="$(printf '%s\n' "${facts}" | sed -n 's/^cores=//p')"
	model="$(printf '%s\n' "${facts}" | sed -n 's/^model=//p')"
	info "  ${model:-unknown board} -- ${arch}, ${cores} core(s)"

	if [ "${arch}" != "${PI_REQUIRED_ARCH}" ]; then
		error "this board is ${arch}, but ${PI_REQUIRED_ARCH} is required."
		error "There is no apache-tvm wheel for ${arch}, and the native build"
		error "targets aarch64. Use a 64-bit Raspberry Pi OS on a Pi 4."
		return 1
	fi
	return 0
}

install_pi_deps() {
	info "installing apache-tvm==${PI_TVM_VERSION} on the Pi (once; a few minutes)"
	# The [rpc] extra is not optional here: tvm.exec.rpc_server imports tornado,
	# and the server dies on startup without it. psutil and cloudpickle come
	# along with the same extra and are needed by the measurement worker.
	#
	# Plain install first: a private interpreter accepts it. Debian's system
	# Python refuses under PEP 668 and needs --break-system-packages.
	if ! ssh_run "${PI_PYTHON} -c 'import tvm, tornado' 2>/dev/null"; then
		ssh_run "${PI_PYTHON} -m pip install -q \
			'apache-tvm[rpc]==${PI_TVM_VERSION}' numpy ||
			${PI_PYTHON} -m pip install -q --break-system-packages \
			'apache-tvm[rpc]==${PI_TVM_VERSION}' numpy" ||
			{
				error "installing apache-tvm on the Pi failed."
				error "Try manually: ${PI_PYTHON} -m pip install \
'apache-tvm[rpc]==${PI_TVM_VERSION}' numpy"
				return 1
			}
	fi

	# A version mismatch between host and Pi breaks the RPC handshake in ways
	# that surface much later as opaque deserialisation errors, so check now.
	# An empty result means the import died -- typically a segfault in dlopen
	# from a half-broken system-Python install; see the note in the header.
	local pi_version
	pi_version="$(ssh_run "${PI_PYTHON} -c 'import tvm; print(tvm.__version__)'" 2>/dev/null || true)"
	if [ "${pi_version}" != "${PI_TVM_VERSION}" ]; then
		error "Pi has TVM '${pi_version}', host has ${PI_TVM_VERSION}."
		error "RPC requires the same version on both ends."
		return 1
	fi
	info "  TVM ${pi_version} ready on the Pi"
	return 0
}

start_pi_rpc_server() {
	local addr="$1"
	local port="$2"
	local key="$3"
	info "starting rpc_server on the Pi"
	ssh_run "pkill -f 'tvm[.]exec[.]rpc_server'; true" >/dev/null 2>&1 || true
	# setsid + nohup so the server survives this SSH session closing.
	ssh_run "setsid nohup ${PI_PYTHON} -m tvm.exec.rpc_server \
		--tracker=${addr}:${port} --key=${key} --port=9090 \
		>/tmp/tvm_rpc_server.log 2>&1 &" >/dev/null 2>&1 || {
		error "failed to launch rpc_server on the Pi"
		return 1
	}
	PI_RPC_STARTED=1
	return 0
}

# Best-effort LAN address; the Pi has to be able to reach us on it.
#
# Ask the routing table which interface carries the default route rather than
# guessing a name: en0 is the built-in port, but a USB or Thunderbolt adapter
# shows up as en8 (or higher), and on those machines guessing en0 yields an
# empty address and an unusable instruction line.
host_address() {
	local addr iface
	iface="$(route -n get default 2>/dev/null |
		awk '/interface:/{print $2; exit}')"
	if [ -n "${iface}" ]; then
		addr="$(ipconfig getifaddr "${iface}" 2>/dev/null || true)"
	fi
	if [ -z "${addr:-}" ]; then
		# Linux: first global address on the default route.
		addr="$(ip route get 1.1.1.1 2>/dev/null |
			awk '/src/{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')"
	fi
	if [ -z "${addr:-}" ]; then
		addr="$(hostname -I 2>/dev/null | awk '{print $1}')"
	fi
	printf '%s' "${addr:-<this-machine-ip>}"
}

start_tracker() {
	local port="$1"
	info "starting RPC tracker on port ${port}"
	"${VENV_PYTHON}" -m tvm.exec.rpc_tracker \
		--host 0.0.0.0 --port "${port}" >/tmp/tvm_tracker.log 2>&1 &
	TRACKER_PID=$!
	sleep 3
	if ! kill -0 "${TRACKER_PID}" 2>/dev/null; then
		error "tracker failed to start; see /tmp/tvm_tracker.log"
		exit 1
	fi
}

print_pi_instructions() {
	local addr="$1"
	local port="$2"
	local key="$3"
	printf '\n'
	info "$(color_bold)Run this ON THE PI, in another terminal:$(color_reset)"
	printf '%s\n' "$(color_cyan)  ${PI_PYTHON} -m tvm.exec.rpc_server \\$(color_reset)"
	printf '%s\n' "$(color_cyan)    --tracker=${addr}:${port} \\$(color_reset)"
	printf '%s\n' "$(color_cyan)    --key=${key} --port=9090$(color_reset)"
	printf '\n'
}

wait_for_device() {
	local port="$1"
	local key="$2"
	info "waiting for '${key}' to register with the tracker..."
	local i
	for i in $(seq 1 120); do
		if "${VENV_PYTHON}" - "$port" "$key" <<-'PY' 2>/dev/null; then
			import sys
			from tvm import rpc
			port, key = int(sys.argv[1]), sys.argv[2]
			summary = rpc.connect_tracker("127.0.0.1", port).text_summary()
			sys.exit(0 if key in summary else 1)
		PY
			info "$(color_green)device '${key}' is registered$(color_reset)"
			return 0
		fi
		sleep 5
	done
	error "timed out waiting for '${key}'. Is the rpc_server running on the Pi,"
	error "and can the Pi reach this machine on port ${port}?"
	return 1
}

main() {
	local pi_host=""
	local key="rpi4"
	local trials=3000
	local port=9190
	local onnx="${REPO_ROOT}/models/onnx/yolov8n.onnx"
	local do_wait=1

	while [ $# -gt 0 ]; do
		case "$1" in
		-p | --pi)
			shift
			pi_host="${1:?--pi requires a hostname}"
			;;
		-s | --ssh)
			shift
			SSH_TARGET="${1:?--ssh requires user@host}"
			;;
		--pi-python)
			shift
			PI_PYTHON="${1:?--pi-python requires a path}"
			;;
		-k | --key)
			shift
			key="${1:?--key requires a name}"
			;;
		-t | --trials)
			shift
			trials="${1:?--trials requires a number}"
			;;
		--port)
			shift
			port="${1:?--port requires a number}"
			;;
		--onnx)
			shift
			onnx="${1:?--onnx requires a path}"
			;;
		--no-wait) do_wait=0 ;;
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

	# --ssh carries the host, so accept it as the --pi value too.
	if [ -z "${pi_host}" ] && [ -n "${SSH_TARGET}" ]; then
		pi_host="${SSH_TARGET#*@}"
	fi
	if [ -z "${pi_host}" ]; then
		error "--pi is required"
		show_help
		exit 1
	fi
	if [ ! -x "${VENV_PYTHON}" ]; then
		error "conversion venv missing: ${VENV_PYTHON}"
		error "Create it per models/README.md."
		exit 1
	fi
	if [ ! -f "${onnx}" ]; then
		error "model not found: ${onnx}"
		exit 1
	fi

	# Verify the board before starting anything, so a wrong target fails in
	# seconds instead of after the tracker and an install are already running.
	if [ -n "${SSH_TARGET}" ]; then
		preflight_pi || exit 1
	fi

	trap cleanup EXIT

	local addr
	addr="$(host_address)"
	start_tracker "${port}"

	if [ -n "${SSH_TARGET}" ]; then
		install_pi_deps || exit 1
		start_pi_rpc_server "${addr}" "${port}" "${key}" || exit 1
	else
		print_pi_instructions "${addr}" "${port}" "${key}"
	fi

	if [ "${do_wait}" -eq 1 ]; then
		wait_for_device "${port}" "${key}" || exit 1
	fi

	info "$(color_bold)tuning against ${pi_host} (${trials} trials)$(color_reset)"
	warn "This measures every candidate on the Pi, so it is much slower than"
	warn "tuning locally. Expect several hours. Leave the Pi powered and cool."

	# --target arm64 pins mtriple/mcpu/num-cores to the Pi 4, so the search
	# optimises for the machine that will actually run the model.
	# Allow TVM/XGBoost OpenMP coexistence; see models/README.md for limitations.
	KMP_DUPLICATE_LIB_OK=TRUE \
		"${VENV_PYTHON}" "${REPO_ROOT}/models/convert/tvm_compile.py" \
		--onnx "${onnx}" \
		--input-name images --input-shape 1,3,320,320 \
		--out yolov8n_tuned \
		--target arm64 \
		--emit-embed "${GENERATED_DIR}" \
		--tune \
		--tune-dir "${REPO_ROOT}/build-tvm/tune-rpi4" \
		--tune-trials "${trials}" \
		--rpc-host 127.0.0.1 --rpc-port "${port}" --rpc-key "${key}"

	printf '\n'
	info "$(color_green)tuned sources written to ${GENERATED_DIR}$(color_reset)"
	info "Now build the artifact:"
	printf '%s\n' "$(color_cyan)  ./scripts/build-pi-artifacts.sh --mode native$(color_reset)"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
	main "$@"
fi
