# Linux artifact builds

[Dockerfile.crossbuild](Dockerfile.crossbuild) builds a Linux application and shared library for `aarch64` (Raspberry Pi 4) or `x86_64`. Docker with Buildx supplies the Linux toolchain on Windows, macOS, or Linux. Shell-wrapper commands require Bash; use Git Bash or WSL on Windows.

## Build

Run from the repository root. Prepare the [model inputs](../models/README.md) before building a real detector.

```bash
docker buildx build --file docker/Dockerfile.crossbuild --target artifact \
  --build-arg TARGET_ARCH=aarch64 --build-arg MODE=native \
  --build-arg MODEL=real --output type=local,dest=dist/pi4-native .
```

For ONNX mode:

```bash
docker buildx build --file docker/Dockerfile.crossbuild --target artifact \
  --build-arg TARGET_ARCH=aarch64 --build-arg MODE=onnx \
  --output type=local,dest=dist/pi4-onnx .
cp models/onnx/yolov8n.onnx dist/pi4-onnx/
```

**Direct Docker builds do not copy the ONNX model.** The wrapper `./scripts/build-pi-artifacts.sh` builds both modes and copies an available ONNX file afterward. A missing ONNX file produces a warning, not a failed build.

Use `MODEL=placeholder` for a native development artifact without TVM/model dependencies. It reports no detections. Use `TARGET_ARCH=x86_64` to build for a Linux desktop; x86_64 SIMD builds require AVX2/FMA-capable hardware.

In PowerShell, put commands on one line or replace Bash line continuations with PowerShell backticks.

## Build arguments

| Argument              | Default   | Values / purpose                           |
| --------------------- | --------- | ------------------------------------------ |
| `TARGET_ARCH`         | `aarch64` | `aarch64`, `x86_64`                        |
| `MODE`                | `native`  | `native`, `onnx`                           |
| `MODEL`               | `auto`    | Native only: `auto`, `real`, `placeholder` |
| `SCALAR_KERNELS`      | `OFF`     | Native only: `ON` for reference kernels    |
| `BUILD_TYPE`          | `Release` | `Release`, `Debug`                         |
| `ONNXRUNTIME_VERSION` | `1.18.1`  | ONNX Runtime archive version               |
| `TVM_GIT_TAG`         | `v0.26.0` | TVM runtime source revision                |

`MODEL=real` builds the static TVM runtime and requires the generated kernels and blob. `auto` does not provision that runtime; use `real` explicitly for deployable native detection.

The build stage runs on `$BUILDPLATFORM`. It installs a cross-compiler when host and target architectures differ; no QEMU registration is needed. Tests run only when the target can execute on the build host. Cross-built binaries still need tests on their target.

## Output and deployment

```text
dist/pi4-native/
  cctv_app
  libcctv.so
  static/
  include/cctv/cctv.h
  LICENSE
  ARCH.txt
```

ONNX artifacts also contain `libonnxruntime.so*` and need the model copied as shown above. Real native artifacts link the TVM runtime and model into `libcctv.so`. No Python, Docker, or TVM installation is needed on the Pi. System glibc and ALSA libraries are still required.

```bash
cat dist/pi4-native/ARCH.txt
file dist/pi4-native/libcctv.so
scp -r dist/pi4-native pi@raspberrypi.local:~/
ssh pi@raspberrypi.local 'cd ~/pi4-native && ./cctv_app --port 8080'
```

The image checks the library's target architecture and records both binary formats. Keep the entire directory together. For device access, tokens, and troubleshooting, see the [root README](../README.md).

## Appendix: portability decisions

- **Bullseye base:** building against an older supported glibc reduces the minimum system-library requirement. The base is digest-pinned; upgrading it requires checking target compatibility, not just whether the image builds.
- **Archived security packages:** Bullseye security packages are fetched from `archive.debian.org` because the original security repository returns 404 for archived packages. Expiry checking is disabled only for this frozen repository; signature and package-hash verification remain enabled. This preserves the older build environment but does not provide ongoing security updates.
- **Static C++ runtime:** `CCTV_PORTABLE_RUNTIME=ON` removes a dynamic libstdc++/libgcc dependency. It does not make glibc static or certify a particular `GLIBC_*` floor. Verify both binaries and bundled libraries on each supported OS release.
- **Native compiler execution:** cross-compilation avoids emulating the compiler while still generating target binaries.
- **Target CPU:** aarch64 cross-build SIMD flags use `cortex-a72`, not the workstation's CPU. Local aarch64 builds use `native` unless overridden.
- **Target ALSA:** multiarch headers/libraries keep microphone support linked for the target rather than accidentally using host libraries.
- **Relative loader paths:** `$ORIGIN` allows binaries and bundled dependencies to remain together after copying without setting `LD_LIBRARY_PATH`.
