# Model export and compilation

These offline tools produce `yolov8n.onnx` for ONNX Runtime and embedded
YOLOv8n kernels/weights for the native backend. Conversion runs on the build
host; remote tuning additionally measures candidates on the Raspberry Pi.
There is no audio model.

## Files

| Path | Purpose | In Git |
| --- | --- | --- |
| [convert/](convert/) | TVM compiler, synthetic fixture generator, dependencies | Yes |
| [download.sh](download.sh) | Local ONNX export or URL download | Yes |
| [onnx/SHA256SUMS](onnx/SHA256SUMS) | Reference digest for downloaded ONNX files | Yes |
| `onnx/yolov8n.onnx` | Exported detection model | No |
| [tuning/rpi4/](tuning/rpi4/) | Pi 4 MetaSchedule workload and measurement database | Yes |
| `../src/backend_native/generated/` | Generated object and weight blob | No |

## Setup and export

Use a dedicated Python environment. The dependencies include PyTorch and TVM
and require several GB of disk space. Dependency wheel availability depends on
the host OS, architecture, and Python version; do not assume every host has a
compatible TVM wheel. The compiler targets the TVM 0.26 Relax API.

```bash
python3 -m venv .venv-convert
.venv-convert/bin/pip install -r models/convert/requirements.txt
./models/download.sh
```

The default export uses Ultralytics weights, `imgsz=320`, opset 13, static shapes,
and no embedded NMS. The input size must match
[`kInputSize`](../src/core/image_ops.hpp); post-processing expects `[1,84,2100]`.

For a hosted export, set `YOLO_ONNX_URL` and run `./models/download.sh --fetch`.
A matching entry in `SHA256SUMS` enables digest verification; a mismatch deletes
the downloaded file. Missing or placeholder entries only warn and skip the
check. Local exports are not checked against this reference digest and may
differ byte-for-byte across exporter versions.

## Compile the native Pi model

Apply the committed Pi 4 tuning database rather than repeating measurements:

```bash
.venv-convert/bin/python models/convert/tvm_compile.py \
  --onnx models/onnx/yolov8n.onnx \
  --input-name images --input-shape 1,3,320,320 \
  --out yolov8n_tuned --target arm64 \
  --emit-embed src/backend_native/generated \
  --tune-db models/tuning/rpi4
```

This writes `yolov8n_tuned_kernels.o` and `yolov8n_tuned_blob.cc`.
Pass `--input-shape 1,3,320,320` explicitly: the compiler's legacy default is
`1,3,640,640`, which does not match this library.
The object is target-specific: do not link an aarch64 object into a host build
for another architecture. CMake also needs a TVM runtime; the
[Docker build](../docker/README.md) builds and links it statically for
`MODE=native MODEL=real`.

Inspect `applied tuned schedules: ... parallel, ... vectorized loops`.
A low count warrants checking graph/database compatibility; completing a
compile does not prove that useful schedules were applied.

Numeric comparison against ONNX Runtime runs only for non-embedded `host`
compilation unless disabled. Cross-target and embedded exports skip this check;
validate detections on the target before using an artifact.

## Re-tune for a different target

```bash
./scripts/tune-on-pi.sh --ssh <user>@<pi-host> \
  --pi-python '$HOME/.local/cctv-tune/python/bin/python3.11'
```

The script requires an aarch64 board and reachable RPC tracker/server.
Candidates are built on the workstation and timed on the Pi.
Allow hours for measurement. Review the resulting `database_*.json` before
replacing the committed database. Changing hardware, graph, or input size may
require new schedules.

## Test fixture

[make_test_models.py](convert/make_test_models.py) produces the small committed
ONNX fixture in [tests/fixtures/onnx/](../tests/fixtures/onnx/). Its fixed
detections test decoding and API integration, not YOLO accuracy.

## Appendix: conversion and linking decisions

- **Simplify before importing:** static shapes are needed by Relax lowering.
  A failed simplification currently logs a warning and retains the original graph.
- **Keep weights constant:** `keep_params_in_input=False` embeds initializers and
  avoids dynamic operands in the imported YOLO graph.
- **Explicit lowering:** `LegalizeOps`, `AnnotateTIROpPattern`, `FoldConstant`,
  `FuseOps`, and `FuseTIR` lower high-level operators to fused kernels.
- **Explicit database application:** both new and reused tuning databases go
  through `MetaScheduleApplyDatabase` before compilation. This preserves the
  selected schedules and makes the annotation count observable.
- **LLVM object, not C kernels:** tuned vector and parallel operations need LLVM
  code generation. The generated blob carries weights and Relax VM bytecode.
- **System-library registration:** the module attribute and packed imports
  register embedded symbols, avoiding runtime model-file loading.
- **Whole-archive and pthreads:** static TVM registration objects must survive
  linking, and parallel schedules require the thread runtime.
- **Cached VM:** native inference reuses a process-wide VM. Initialization is
  thread-safe; concurrent inference calls into that VM are not guaranteed safe.
- **Persist measurements, regenerate binaries:** tuning costs much more than
  compilation. Commit the database, not model weights or generated objects.
- **Remote OpenMP setting:** the tuning wrapper sets `KMP_DUPLICATE_LIB_OK` for
  TVM/XGBoost runtime coexistence. This is a tooling workaround, not a general
  guarantee that mixing OpenMP runtimes is safe.

### Historical Pi measurement

The previous project notes recorded a Pi 4 Model B (4 GB), Debian 13, a USB
camera at 640x480, and a database built from 3093 measured candidates:

| Mode | Mean pipeline fps | Observed range |
| --- | --- | --- |
| `onnx` | 6.13 | 5.96-6.29 |
| `native` | 5.97 | 5.76-6.11 |

These are retained historical notes, not results reproduced by CI or this
documentation update. They summarize ten `/status` samples across two runs per
mode and include capture, inference, annotation, and JPEG encoding. Raw samples
are not committed. Thermal throttling and starting temperature limit comparison;
do not treat these values as a performance guarantee.
