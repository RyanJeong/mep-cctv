#!/usr/bin/env bash
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# ============================================================================
# models/download.sh
# ============================================================================
# Usage:
#   ./models/download.sh [OPTIONS]
#
# Description:
#   Produce models/onnx/yolov8n.onnx, which the onnx mode needs.  By default
#   the model is exported locally with Ultralytics. ONNX mode loads the file;
#   native model compilation uses it as an offline input.
#
#   --fetch downloads instead, and only works against a URL you host: set
#   YOLO_ONNX_URL to point at it.
#
#   Checksum verification: after the download, the SHA-256 digest is compared
#   against a populated entry in models/onnx/SHA256SUMS. A mismatch deletes
#   the file and exits non-zero; missing entries warn and skip verification.
#
#   URL override: set YOLO_ONNX_URL in the environment to point at a different
#   source.
#
# Options:
#   -h, --help       Show this help and exit
#   -f, --fetch      Download instead of exporting (needs YOLO_ONNX_URL)
#   -v, --venv PATH  Virtualenv to export with (default: .venv-convert)
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers (printf '\033[Xm' pattern; ASCII only; no Unicode)
# ---------------------------------------------------------------------------
color_red() { printf '\033[31m'; }
color_green() { printf '\033[32m'; }
color_yellow() { printf '\033[33m'; }
color_reset() { printf '\033[0m'; }

info() { printf '%s\n' "$(color_green)[INFO]$(color_reset) ${1:-}"; }
warn() { printf '%s\n' "$(color_yellow)[WARN]$(color_reset) ${1:-}"; }
error() { printf '%s\n' "$(color_red)[ERROR]$(color_reset) ${1:-}" >&2; }

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ONNX_DIR="${SCRIPT_DIR}/onnx"

readonly YOLO_ONNX="${ONNX_DIR}/yolov8n.onnx"

# Pre-exported ONNX download URL, overridable before calling this script:
#   YOLO_ONNX_URL=https://... bash models/download.sh
#
# The default URL is a placeholder; --fetch needs a hosted export URL.
# Digest verification requires a populated SHA256SUMS entry.
YOLO_ONNX_URL="${YOLO_ONNX_URL:-https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx}"
readonly YOLO_ONNX_URL

# SHA256SUMS file location
readonly SHA256SUMS_FILE="${ONNX_DIR}/SHA256SUMS"

# Placeholder sentinel -- presence means hashes have not been filled in yet
readonly SHA256_PLACEHOLDER="PLACEHOLDER_FILL_IN_REAL_SHA256"

# Default option values
MODE="export"
VENV_PATH=".venv-convert"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
        -h | --help)
            show_help
            exit 0
            ;;
        -f | --fetch)
            MODE="fetch"
            shift
            ;;
        -v | --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

check_downloader() {
    if command -v curl >/dev/null 2>&1; then
        echo "curl"
    elif command -v wget >/dev/null 2>&1; then
        echo "wget"
    else
        error "Neither curl nor wget is available. Install one and retry."
        exit 1
    fi
}

download_file() {
    local url="$1"
    local dest="$2"
    local tool
    tool="$(check_downloader)"

    if [ -f "$dest" ]; then
        info "Already exists (skipping): $(basename "$dest")"
        return 0
    fi

    info "Downloading $(basename "$dest") ..."
    case "$tool" in
    curl) curl --fail --location --progress-bar --output "$dest" "$url" ;;
    wget) wget --quiet --show-progress --output-document "$dest" "$url" ;;
    esac
    info "Saved: $dest"
}

# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------

# Portable SHA-256: prefer sha256sum (Linux/coreutils), fall back to shasum
# (macOS).  Prints the hex digest only; exits non-zero if unavailable.
sha256_of() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "$file" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -- "$file" | awk '{print $1}'
    else
        error "No sha256sum or shasum found -- cannot verify checksums."
        exit 1
    fi
}

# Verify a downloaded file against SHA256SUMS.
# $1 = path to the downloaded file (absolute or relative)
# Behaviour:
#   - If SHA256SUMS does not exist: warn and skip (setup not yet done).
#   - If the entry for this file contains the placeholder sentinel: warn +
#     skip (maintainer has not filled in real hashes yet).
#   - If the digest matches: print info + return 0.
#   - If the digest mismatches: delete the file, print error, exit 1.
verify_checksum() {
    local dest="$1"
    local filename
    filename="$(basename -- "$dest")"

    if [ ! -f "$SHA256SUMS_FILE" ]; then
        warn "SHA256SUMS not found -- skipping integrity check for ${filename}"
        return 0
    fi

    local expected
    expected=$(grep -- "${filename}" "$SHA256SUMS_FILE" 2>/dev/null | awk '{print $1}' || true)

    if [ -z "$expected" ]; then
        warn "No entry for ${filename} in SHA256SUMS -- skipping integrity check"
        return 0
    fi

    if [ "$expected" = "$SHA256_PLACEHOLDER" ]; then
        warn "SHA256SUMS still contains placeholder for ${filename}."
        warn "Fill in the real hash in ${SHA256SUMS_FILE} before production use."
        return 0
    fi

    local actual
    actual=$(sha256_of "$dest")

    if [ "$actual" = "$expected" ]; then
        info "Checksum OK: ${filename} (${actual})"
    else
        error "Checksum MISMATCH for ${filename}:"
        error "  expected: ${expected}"
        error "  actual:   ${actual}"
        rm -f -- "$dest"
        error "Deleted partial/tampered file: ${dest}"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

run_fetch() {
    mkdir -p "$ONNX_DIR"
    info "Fetching pre-exported ONNX files into $ONNX_DIR"
    download_file "$YOLO_ONNX_URL" "$YOLO_ONNX"
    verify_checksum "$YOLO_ONNX"
    info "Done."
}

# Export locally; --fetch instead uses a separately hosted ONNX file.
run_export() {
    local python_bin
    python_bin="$(locate_python)"
    # Resolve to an absolute path before the subshell cd: locate_python may
    # return a relative venv path OR the bare command "python3", and only an
    # absolute path survives changing directory.
    if [ -x "$python_bin" ]; then
        python_bin="$(cd "$(dirname "$python_bin")" && pwd)/$(basename "$python_bin")"
    else
        python_bin="$(command -v "$python_bin")"
    fi
    mkdir -p "$ONNX_DIR"
    info "Exporting yolov8n.onnx with $python_bin"

    # imgsz must match kInputSize in src/core/image_ops.hpp, and the anchor
    # count derived from it in yolo_postprocess.hpp. Exporting at a different
    # size produces a model that loads fine and then decodes to nonsense.
    (
        cd "$ONNX_DIR" && "$python_bin" - <<'PY'
from ultralytics import YOLO

YOLO("yolov8n.pt").export(format="onnx", imgsz=320, opset=13,
                          simplify=False, dynamic=False)
PY
    ) || {
        error "Export failed. Is ultralytics installed in the venv?"
        exit 1
    }

    [ -f "$YOLO_ONNX" ] || {
        error "Export produced no $YOLO_ONNX"
        exit 1
    }

    # Deliberately not verified against SHA256SUMS: ONNX export is not
    # bit-reproducible, so a local export differs from the recorded digest even
    # on the exact versions it was recorded with. That digest exists to
    # validate a *download*. What matters downstream is the graph, not the
    # bytes -- the tuning database keys on the lowered TIR, so it applies to
    # any export of the same model at the same size.
    info "Exported $(basename "$YOLO_ONNX") ($(wc -c <"$YOLO_ONNX" | tr -d ' ') bytes)"
    info "Done."
}

locate_python() {
    local venv_python="${VENV_PATH}/bin/python"
    if [ -f "$venv_python" ]; then
        echo "$venv_python"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return 0
    fi
    error "No Python found. Activate a venv or set --venv."
    exit 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    parse_args "$@"

    info "mep-cctv model download/export script"
    info "Mode   : $MODE"

    case "$MODE" in
    fetch) run_fetch ;;
    export) run_export ;;
    *)
        error "Unknown mode: $MODE"
        exit 1
        ;;
    esac

    info "All done."
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
