# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-17

Initial release of an educational CCTV application for Raspberry Pi 4. It compares ONNX Runtime inference with a TVM-compiled version of the same YOLOv8n model and includes a development-only native placeholder mode.

### Added

- Shared C API in `include/cctv/cctv.h` with mutually exclusive `onnx` and `native` build modes; native builds can use a TVM-generated model or a zero-detection placeholder.
- C++ application pipeline for V4L2 video and optional ALSA audio capture, with synthetic video and silent audio fallbacks when devices are unavailable.
- HTTP application server with an MJPEG stream at `/stream`, JSON metrics at `/status`, a health endpoint at `/healthz`, and optional token authentication for the stream and status endpoints.
- COCO-labelled detection annotations with confidence scores, an embedded bitmap font, and a colour palette for distinguishing boxes.
- Sound-activity detection using a DC-corrected RMS calculation; audio is processed without an audio model or person classifier.
- Scalar reference and SIMD implementations of image resizing and non-maximum suppression, with native scalar-kernel builds available for comparison.
- Host metrics for CPU usage, resident memory, and CPU temperature when available on Linux.
- Model export and TVM compilation utilities, including a Raspberry Pi 4 tuning database and a script for tuning remotely against Pi hardware.
- Raspberry Pi artifact build and cleanup scripts, plus benchmark utilities for comparing latency, peak memory, and library size when real model artifacts are supplied.
- C++ and Python test suites, along with CI coverage for ONNX, native SIMD, native scalar, sanitizers, benchmarks, and repository checks.
- Documentation for local builds, Raspberry Pi deployment, model preparation, benchmarks, and known design and deployment limitations.

[0.1.0]: https://github.com/RyanJeong/mep-cctv/releases/tag/v0.1.0
