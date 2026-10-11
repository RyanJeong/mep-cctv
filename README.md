# mep-cctv

[![CI](https://github.com/RyanJeong/mep-cctv/actions/workflows/ci.yml/badge.svg)](https://github.com/RyanJeong/mep-cctv/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/RyanJeong/mep-cctv/graph/badge.svg)](https://codecov.io/gh/RyanJeong/mep-cctv)

An educational CCTV application for Raspberry Pi 4. It compares ONNX Runtime inference with a TVM-compiled version of the same YOLOv8n model.

## What it does

- Captures video, detects objects, and draws COCO labels and confidence scores.
- Serves annotated frames as MJPEG at `/stream` and metrics as JSON at `/status`.
- Detects sound using DC-corrected RMS, not an audio model or person classifier.
- Reports process CPU usage, resident memory, and CPU temperature on Linux.
- Uses synthetic video and silent audio when capture devices are unavailable.

Both inference modes expose the same [C API](include/cctv/cctv.h):

| Mode                        | Model execution                                     | Runtime files for a Pi deployment                    |
| --------------------------- | --------------------------------------------------- | ---------------------------------------------------- |
| `onnx`                      | ONNX Runtime loads the graph                        | `libcctv.so`, ONNX Runtime libraries, `yolov8n.onnx` |
| `native` with a real model  | TVM kernels and weights are linked into the library | `libcctv.so` with a statically linked TVM runtime    |
| `native` with a placeholder | No inference; returns no detections                 | `libcctv.so`; for development only                   |

The model input is RGB float32 NCHW at **320x320**. Exported models must match [`kInputSize`](src/core/image_ops.hpp) and the YOLOv8 output layout `[1,84,2100]`.

## Project layout

| Path                        | Purpose                                                        |
| --------------------------- | -------------------------------------------------------------- |
| [include/](include/)        | Public C API                                                   |
| [src/](src/)                | API implementation, inference backends, image/audio processing |
| [app/](app/)                | Capture, annotation, HTTP server, and web assets               |
| [models/](models/README.md) | Export, TVM compilation, and committed Pi tuning data          |
| [bench/](bench/)            | Python benchmarks and resource-monitoring script               |
| [tests/](tests/)            | C++ tests and synthetic ONNX fixture                           |
| [docker/](docker/README.md) | Linux cross-build image and artifact layout                    |
| [scripts/](scripts/)        | Build, cleanup, artifact, and remote-tuning helpers            |

## Local build

Requires CMake >= 4.0, a GCC/Clang C++14 toolchain, and network access for GoogleTest, cpp-httplib, and stb. TVM integration uses C++17. On Linux, install ALSA development headers/libraries if microphone capture is needed.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCCTV_MODE=native -DCCTV_MODEL=placeholder -DBUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/cctv_app --host 127.0.0.1 --port 8080 --static app/static
```

Open `http://localhost:8080/`. This build tests the pipeline, **not object detection**. No model download or TVM installation is required, but build dependencies are fetched during configuration.

For an ONNX build, export the model first and configure with `-DCCTV_MODE=onnx`. CMake can fetch the pinned Linux ONNX Runtime archive; to supply your own installation, set `-DONNXRUNTIME_ROOT=/path/to/onnxruntime`. Run with `--yolo models/onnx/yolov8n.onnx`.

### Build options

| Option                        | Default  | Meaning                                       |
| ----------------------------- | -------- | --------------------------------------------- |
| `CCTV_MODE`                   | `native` | `native` or `onnx`                            |
| `CCTV_MODEL`                  | `auto`   | Native only: `auto`, `real`, or `placeholder` |
| `CCTV_SCALAR_KERNELS`         | `OFF`    | Native only: use scalar reference kernels     |
| `BUILD_APP` / `BUILD_TESTS`   | `ON`     | Build the server / unit tests                 |
| `ENABLE_COVERAGE`             | `OFF`    | Instrument coverage                           |
| `ENABLE_ASAN` / `ENABLE_TSAN` | `OFF`    | ASan+UBSan / TSan; mutually exclusive         |
| `CCTV_PORTABLE_RUNTIME`       | `OFF`    | Linux only: statically link the C++ runtime   |

`auto` selects the real native model only when generated files and an explicit TVM runtime location are present. `real` requires those dependencies. ONNX mode resets native-only options to their defaults. Legacy option names are translated and removed from the CMake cache.

## Build and deploy to Raspberry Pi 4

Use a 64-bit Raspberry Pi OS (`uname -m` must report `aarch64`), a V4L2 USB camera, and optionally an ALSA microphone. Provide adequate power and cooling. The build host needs Docker with Buildx and a Python conversion environment. See [model setup](models/README.md) for dependency and tuning details.

```bash
python3 -m venv .venv-convert
.venv-convert/bin/pip install -r models/convert/requirements.txt
./models/download.sh
.venv-convert/bin/python models/convert/tvm_compile.py \
  --onnx models/onnx/yolov8n.onnx \
  --input-name images --input-shape 1,3,320,320 \
  --out yolov8n_tuned --target arm64 \
  --emit-embed src/backend_native/generated \
  --tune-db models/tuning/rpi4
./scripts/build-pi-artifacts.sh
scp -r dist/pi4-onnx dist/pi4-native pi@raspberrypi.local:~/
```

The wrapper currently checks for native generated files even with `--mode onnx`. For an ONNX-only build without them, use `--mode onnx --placeholder`; this flag does not change ONNX inference. Direct Docker builds are documented [in the Docker build guide](docker/README.md).

Keep each artifact folder intact. On the Pi, add your user to `video` and `audio`, then log out and back in:

```bash
sudo usermod -aG video,audio "$USER"
cd ~/pi4-native
./cctv_app --port 8080 --token '<local-access-token>'
```

Open `http://raspberrypi.local:8080/?token=<local-access-token>`. Run one mode at a time to avoid camera and port conflicts. The startup banner identifies the selected backend and capture sources.

### Application options

| Option                 | Default                   | Meaning                                             |
| ---------------------- | ------------------------- | --------------------------------------------------- |
| `--host` / `--port`    | `0.0.0.0` / `8080`        | HTTP bind address / port                            |
| `--video` / `--audio`  | `/dev/video0` / `default` | Capture devices                                     |
| `--width` / `--height` | `640` / `480`             | Capture dimensions, not model dimensions            |
| `--yolo`               | Auto-discovered           | ONNX model path; ignored by native mode             |
| `--static`             | Auto-discovered           | Web assets beside the executable, then `app/static` |
| `--token`              | Unset                     | Require a query token for `/stream` and `/status`   |
| `--quality`            | `80`                      | JPEG quality, 1-100                                 |
| `-h`, `--help`         |                           | Print usage                                         |

The server uses HTTP, not TLS. A query token is not sufficient protection for an untrusted network. Keep this teaching application on a trusted local network; static assets and `/healthz` are not token-protected.

### Troubleshooting

| Symptom                             | Check                                                                                                                                                            |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Wrong executable format             | Target architecture and 64-bit OS                                                                                                                                |
| Missing shared library              | Copy the complete artifact folder                                                                                                                                |
| `synthetic` / `silent` source       | Device availability, permissions, ALSA build support                                                                                                             |
| ALSA device errors                  | Use `arecord -l` and pass an explicit `--audio` device                                                                                                           |
| Falling frame rate                  | Power and thermal status: `vcgencmd get_throttled`                                                                                                               |
| Video works but metrics stay at `-` | Check the browser console and `/status` (including `?token=...` when required) for HTTP or JSON errors; rebuild and deploy both the executable and static assets |
| Slow native inference               | Applied schedules and tuning hardware; see [models](models/README.md)                                                                                            |

## Benchmarks and CI

Install [benchmark dependencies](bench/requirements-dev.txt) in a virtualenv. `python bench/compare.py --real-model` builds each mode separately and writes latency, peak RSS, and library size to `bench/reports/comparison.md`. Supply the ONNX model and compatible generated native model/TVM runtime first. Without `--real-model`, native measures a placeholder and is not an inference performance comparison.

[CI](.github/workflows/ci.yml) checks ONNX, native SIMD, and native scalar builds, sanitizers, coverage, Python benchmarks, document/shell syntax, and dependency inventory/scanning. Real TVM inference and Pi device I/O need separate target validation; placeholder tests cannot certify them. C++ coverage merges per-test Clang/LLVM profiles; Python coverage excludes the test modules themselves. [Learn CI](.github/workflows/learn-ci.yml) is a manual teaching example. [Release](.github/workflows/release.yml) publishes an **x86_64 development-only placeholder library** after CI succeeds on `main`, not a Pi detector.

Pull requests run all nine CI jobs once; push events run CI only on `main` to validate the final merged commit and trigger release publication. Pushes to `dev` and feature branches do not run CI separately, avoiding duplicate checks when opening a PR. Codecov adds project and patch coverage checks. Learn CI is never triggered by a push or pull request. The pinned stb JPEG header is patched in the build directory to use unsigned bit accumulation, preserving sanitizer checks without changing upstream source files.

## License

Project source is licensed under [Apache-2.0](LICENSE). Dependencies and model weights retain their own licenses. Ultralytics YOLOv8 uses AGPL-3.0 (or a separate commercial license); compiling its weights into native code does not remove applicable obligations. Check the licenses and corresponding-source requirements for the exact components you distribute or serve. An open repository alone does not establish compliance for every deployment.

## Appendix: design decisions

- **320x320 input:** reduces model work at the expense of small-object accuracy. Changing it requires matching exports, fixtures, and target tuning.
- **No fabricated detections:** the placeholder returns zero output so a missing real model cannot appear to be a working detector.
- **C API ownership:** callers release returned arrays through `cctv_free_detections`; they do not choose the library's allocator.
- **Class-agnostic NMS:** both implementations suppress overlapping boxes regardless of class. This is current behavior, not per-class suppression.
- **Source-row buffers:** de-interleaving covers the capture width before resize. Heap buffers use that width, and 64-bit source offsets support large frames.
- **Annotation:** clip oversized boxes before drawing borders; embedded bitmap glyphs avoid runtime font files.
- **Audio bias removal:** subtracting the mean prevents a constant microphone offset from being classified as sound. The legacy `person_present` field means sound activity, not evidence of a person.
- **Frame snapshots:** annotation precedes JPEG encoding; separate audio capture and brief snapshot locks keep device reads off the HTTP serving path.
- **Metrics:** CPU percentage is normalized across all cores. Unavailable values become JSON `null`; sampling is throttled to once per second.
- **Bundled paths:** relative loader paths and executable-side asset discovery make copied Pi artifact folders relocatable.
- **Editor build database:** only the canonical `build/` refreshes the root compilation-database link, preventing scratch builds from replacing it.
- **TSan runner layout:** the dedicated Ubuntu CI runner reduces mmap ASLR entropy to 28 bits before TSan runs, avoiding shadow-memory collisions. Restricted containers may still prevent TSan startup; this is not a passing race check.
