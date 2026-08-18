# models/ -- ONNX export and TVM compilation

Offline tooling that produces the two things the project can deploy: an `.onnx` file for the `onnx` mode, and pre-compiled sources linked into `libcctv` for the `native` mode. **None of it runs on the Raspberry Pi.**

There is no audio model. Acoustic detection is a short algorithm in `src/core/audio_detect.cc`, so nothing here concerns audio.

## Layout

```text
models/
  convert/
    tvm_compile.py        ONNX -> TVM, and the embeddable sources
    make_test_models.py   tiny synthetic ONNX fixture for the unit tests
    requirements.txt      heavy conversion dependencies
  onnx/
    SHA256SUMS            integrity check for downloaded models
    yolov8n.onnx          gitignored; exported or downloaded
  tuning/
    rpi4/                 committed autotuning database for the Pi 4
  download.sh             export (or download) yolov8n.onnx at 320x320
```

## Setting up

The conversion dependencies are large and are not needed to build or run the project, so they live in their own virtualenv:

```bash
python3 -m venv .venv-convert
.venv-convert/bin/pip install -r models/convert/requirements.txt
```

## Build procedure

Two modes, two procedures. Each step says why it exists, because most of them look optional and are not.

### Mode 1: `onnx`

The original model, executed by ONNX Runtime. One step.

1. **Get the model.**

   ```bash
   ./models/download.sh
   ```

   Or export it yourself:

   ```python
   from ultralytics import YOLO
   YOLO("yolov8n.pt").export(format="onnx", imgsz=320, opset=13,
                             simplify=False, dynamic=False)
   ```

   *Why 320 and not the default 640:* inference cost scales with the square of the input side, so 640 is roughly 4x the work and was unusable on a Pi 4. `imgsz` must match `kInputSize` in `src/core/image_ops.hpp`, and the anchor count derived from it in `yolo_postprocess.hpp`; changing one without the others produces silent garbage rather than an error.

   *Why it exports rather than downloads:* Ultralytics publishes `.pt` weights, not ONNX, so there is no public pre-exported 320x320 model to fetch. `download.sh --fetch` still exists for a URL you host yourself, and that path verifies the file against `SHA256SUMS` and deletes it on mismatch.

   *Why a local export does not match `SHA256SUMS`:* ONNX export is not bit-deterministic, so the digest differs even on the exact ultralytics and torch versions it was recorded with. Nothing downstream depends on the bytes -- the tuning database keys on the lowered graph, so it applies to any export of this model at this size.

2. **Build.** The `.onnx` is copied into the artifact and found at run time beside the executable.

   ```bash
   ./scripts/build-pi-artifacts.sh --mode onnx
   ```

### Mode 2: `native`

The same model compiled to machine code by TVM and linked into `libcctv`, with no model file and no TVM shared library at run time. Three steps, of which only the middle one is expensive.

1. **Get the model.** Exactly as above -- the native mode starts from the same `.onnx`.

2. **Produce the tuned sources.** The repository already contains the autotuning database, so the normal path is to *apply* it rather than tune again:

   ```bash
   .venv-convert/bin/python models/convert/tvm_compile.py \
     --onnx models/onnx/yolov8n.onnx \
     --input-name images --input-shape 1,3,320,320 \
     --out yolov8n_tuned --target arm64 \
     --emit-embed src/backend_native/generated \
     --tune-db models/tuning/rpi4
   ```

   Takes about two minutes and needs no Pi. Check the applied-schedule line in the output -- see "Verifying the schedules were applied" below.

   To tune from scratch instead -- only needed for different hardware or a different model -- see "Re-tuning" below.

3. **Build.**

   ```bash
   ./scripts/build-pi-artifacts.sh --mode native
   ```

   *Why CMake refuses to continue if step 2 was skipped:* the alternative is falling back to the placeholder model, which returns synthetic detections. A build that silently reports fake objects as real is worse than a build that fails.

## What is committed, and what is regenerated

The distinction is cost, not principle. **Tuning takes hours and its output is small**; everything derived from it is large and takes minutes.

| Artifact                               | Size in git | Cost to reproduce | Committed |
| -------------------------------------- | ----------- | ----------------- | --------- |
| `models/tuning/rpi4/*.json` (database) | 0.6 MB      | ~2.5 h on the Pi  | **yes**   |
| `src/.../yolov8n_tuned_kernels.o`      | 0.3 MB      | ~2 min            | no        |
| `src/.../yolov8n_tuned_blob.cc`        | 15.4 MB     | ~2 min            | no        |
| `models/onnx/yolov8n.onnx`             | 6 MB        | one download      | no        |

The database is committed because it is the only artifact whose cost is measured in hours, and because it is what makes the tuned build reproducible: the same database and the same `.onnx` produce the same schedules on any machine. The generated `.o` and blob stay ignored because 15.7 MB of git history to save two minutes is a bad trade, and every re-tune would add another copy forever.

The database is portable across build hosts but **specific to the target hardware** it was measured on -- see "Schedules are specific to the machine they were tuned on".

## Re-tuning

Only needed for different target hardware, a different model, or a different input size.

```bash
./scripts/tune-on-pi.sh --ssh <user>@<pi-host> \
  --pi-python '$HOME/.local/cctv-tune/python/bin/python3.11'
```

Candidates are compiled on the workstation and **measured on the Pi**, because the best loop tiling depends on cache size and core count. Expect roughly 2.5 hours. Copy the resulting `database_*.json` over `models/tuning/rpi4/` to make the new schedules the committed default.

**The script refuses to run against a board that is not `aarch64`**.

## Verifying the schedules were applied

Every compile logs how many tuned loop annotations survived into the final module:

```text
applied tuned schedules: 87 parallel, 158 vectorized loops
```

A near-zero count means the database was not applied and the build is effectively untuned, despite finishing without error. This is the single most useful line in the output; see "compile_relax silently ignores the tuning database" for why it exists.

## Three non-obvious requirements

All three are load-bearing; the conversion fails without them.

1. **Simplify the ONNX graph first** (`onnxsim`). A stock Ultralytics export carries `Shape`/`Slice`/`Concat` plumbing that leaves symbolic shapes, and Relax's `LegalizeOps` rejects tensors without a static `ShapeExpr`.
2. **`keep_params_in_input=False`.** With `True`, initialisers become function parameters, and TVM then reads YOLOv8's `Split` operand as dynamic and raises "Dynamic Split not yet supported". With `False` they stay constants -- which is also what bakes the weights into the artifact.
3. **Run the lowering pipeline explicitly**: `LegalizeOps -> AnnotateTIROpPattern -> FoldConstant -> FuseOps -> FuseTIR`. The default pipeline leaves high-level Relax operators that the code generator cannot lower.

## What `--emit-embed` writes

Two files, both compiled straight into `libcctv`, so the native mode opens no file at run time:

| File               | Contents                                                    |
| ------------------ | ----------------------------------------------------------- |
| `<name>_kernels.o` | The fused kernels, as a prebuilt aarch64 relocatable object |
| `<name>_blob.cc`   | Weights and Relax VM bytecode as a byte array               |

*Why a `.o` rather than C source:* TVM can emit plain C, which looks ideal for embedding, but its C backend cannot express the vector types or `T.parallel` that tuning produces -- it fails with `Cannot convert type float32x20 to C type`. The LLVM backend emits an object file instead, and CMake links a `.o` listed as a source directly, so the result is still a single shared library.

*Why the module is marked a system library:* `mod.with_attr("system_lib_prefix", "")` makes the kernels self-register with TVM's registry at load time. Without it the linked-in object reports `__tvm_is_system_module = False` and the VM segfaults resolving kernels.

## Optimizations applied to the native build

The `native` mode is not just "the model compiled to code" -- unoptimised generated code is roughly **1000x slower** than a tuned inference runtime, which is unusable. Everything below is what closes that gap, in the order it matters.

### 1. Autotuning (MetaSchedule) -- the one that matters

TVM's default pipeline emits **naive loop nests**. The lowered TIR for YOLOv8n contains:

```text
parallel loop annotations   : 0
vectorized loop annotations : 0
total serial loops          : 268
```

No tiling, no vectorization, no threading. Autotuning searches for a better loop arrangement for each of the 66 fused kernels: how to tile them so working sets fit in cache, which loop to vectorise, which to spread across cores, and how much to unroll. A real candidate the search produced:

```python
sch.split(loop=l5, factors=[v12, v13, v14, v15])   # tile into cache blocks
sch.reorder(l16, l24, l32, l40, l17, l25, ...)     # order for locality
sch.annotate(b4, "meta_schedule.parallel",  ann_val=80)
sch.annotate(b4, "meta_schedule.vectorize", ann_val=32)   # this is SIMD
sch.annotate(b4, "meta_schedule.unroll_explicit", ...)
```

Measured on YOLOv8n at 320x320:

| Build     | Latency   |
| --------- | --------- |
| untuned   | 1015.0 ms |
| autotuned | 21.5 ms   |

**47x**, and it is the difference between the native mode being a curiosity and being usable.

Autotuning is **not** retraining. It never touches the weights; the numerical output is unchanged. What it changes is how the same arithmetic is scheduled onto the hardware. There is machine learning inside it -- an XGBoost cost model that predicts runtime -- but the thing being trained is the predictor, not YOLOv8.

Run it with `./scripts/tune-on-pi.sh`, or `tvm_compile.py --tune`.

### 2. SIMD in the pre/post-processing kernels

`src/core/image_ops_simd.cc` and `nms_simd.cc` are hand-written NEON/AVX implementations of letterbox and NMS, selected by `-DCCTV_SIMD=ON`.

Be clear about the scale: these cover roughly **1%** of a frame. They are worth having, they are a good teaching example of intrinsics, and the scalar copies remain as the correctness reference -- but they cannot move the total. The vectorization that matters for speed is the kind autotuning puts inside the model's own loops.

### 3. Input resolution

320x320 rather than YOLOv8's default 640x640. Cost scales with the square of the input, so this alone is about 4x. Accuracy on small and distant objects suffers; people at close-to-medium range still detect reliably.

### 4. Static linking of the TVM runtime

The runtime is built from source as `libtvm_runtime.a` and linked into `libcctv`, so the deployed artifact is a single shared library with no TVM `.so` beside it. Two traps here, both of which fail in confusing ways:

- **Whole-archive is mandatory.** The model publishes its kernels through static initialisers. A normal static link discards them, and the VM then dies with `Global Function ... is not found`. Use CMake's `$<LINK_LIBRARY:WHOLE_ARCHIVE,...>` -- raw `-Wl,--whole-archive` flags get reordered by CMake and silently stop pairing with their archives.
- **pthreads must be linked explicitly.** Without it the tuned `T.parallel` loops still run, but serially. Measured cost of forgetting this: **2.25 fps instead of 9.31**.

## What did NOT work, and why

Documented so nobody spends the time again.

### Quantisation gives TVM nothing

INT8 is the obvious next lever, and for ONNX Runtime it is a real one. For TVM here it is not:

| Build                          | Latency |
| ------------------------------ | ------- |
| TVM fp32                       | 1054 ms |
| TVM int8                       | 1047 ms |
| ONNX Runtime fp32 (one thread) | 32.9 ms |
| ONNX Runtime int8 (one thread) | 10.7 ms |

ORT gains 3.1x because it ships hand-written int8 kernels. TVM gains nothing: an unscheduled loop nest is bound by loop overhead and memory traffic, not arithmetic width, so narrowing the data does not help. The quantised graph also adds 29 more TIR functions for the quantise/dequantise steps.

### The C backend is a dead end for performance

TVM can emit plain C, which looks ideal for embedding. It is not usable for a fast build:

| Variant (one conv2d)                | Latency              | vs naive |
| ----------------------------------- | -------------------- | -------- |
| naive, `target=c`                   | 3344.8 ms            | 1.0x     |
| parallel, `target=c`                | 3346.4 ms            | **1.0x** |
| parallel + vectorize, `target=c`    | **fails to compile** | --       |
| parallel + vectorize, `target=llvm` | 23.9 ms              | **140x** |

The C backend cannot express vector types (`Cannot convert type float32x20 to C type`) and does not realise `T.parallel` either. It is a portability/reference backend. All of TVM's performance lives in the LLVM backend, so the native build embeds an LLVM-generated object rather than C source.

### compile_relax silently ignores the tuning database

`meta_schedule.relax_integration.compile_relax(database, mod, ...)` accepts a database and applies **nothing**, returning a module that still runs the default schedules:

| Path                        | Latency   | Speedup    |
| --------------------------- | --------- | ---------- |
| untuned                     | 1015.0 ms | 1.00x      |
| `MetaScheduleApplyDatabase` | 21.5 ms   | **47.19x** |
| `compile_relax`             | 1069.0 ms | 0.95x      |

This failure mode is silent: hours of tuning appear to produce nothing, and the obvious conclusion -- "autotuning does not help" -- is wrong. `tvm_compile.py` applies the transform directly for this reason.

The trap is worth stating twice, because it bit this project a second time. Fixing `--tune-db` left the identical call in the `--tune` path, where `tune_relax` hands its database straight to `compile_relax`. A full Pi tuning run -- 3093 measured records -- compiled into an artifact byte-for-byte equivalent to an untuned build, with nothing in the log to say so. The tell is the applied-schedule count that `tvm_compile.py` now logs on every compile:

```text
applied tuned schedules: 3 parallel, 4 vectorized loops     <- database ignored
applied tuned schedules: 87 parallel, 158 vectorized loops  <- database applied
```

Check that line after every tuning run. Both paths now go through `compile_with_database`.

### Schedules are specific to the machine they were tuned on

Tuning against a 10-core laptop and running on 4 cores measured **9.31 fps versus 0.88 fps** -- the mismatched parallel decomposition costs more than the core count alone would suggest. Tune for the deployment target: `./scripts/tune-on-pi.sh` measures candidates on the Pi itself.

## Result on the target hardware

Measured on a Raspberry Pi 4 Model B 4GB (Debian 13, USB webcam at 640x480), schedules tuned by measuring 3093 candidates on that same board. Each figure is the mean of ten `/status` samples taken across two runs per mode, alternating and starting from a comparable temperature:

| Mode     | Mean fps | Range     | Per frame |
| -------- | -------- | --------- | --------- |
| `onnx`   | 6.13     | 5.96-6.29 | 163 ms    |
| `native` | 5.97     | 5.76-6.11 | 168 ms    |

The native build reaches **97% of ONNX Runtime**, which is the goal: a model compiled to C/C++ and linked into a single shared library, performing like the vendor runtime it replaces.

### Two caveats worth

This is whole-pipeline throughput -- capture, inference, annotation and JPEG encoding -- not inference alone, so it understates the difference between the two backends. And the board has no active cooling: it reaches 80 C after roughly a minute of sustained load and trips the soft temperature limit, after which frame rate decays a few percent regardless of mode. Compare modes from the same starting temperature, or the second one measured always looks worse.

## AGPL-3.0 warning

YOLOv8 (Ultralytics) is **AGPL-3.0**. The TVM-compiled sources contain YOLOv8n-derived weights and are therefore a derivative work that inherits this license. Distributing a binary containing those weights without offering the complete corresponding source is an AGPL violation.

Because the web page is served over a network, AGPL's network-interaction clause applies: running this as a public service counts as conveying the software, and triggers the same source-offer obligation as shipping a binary.
