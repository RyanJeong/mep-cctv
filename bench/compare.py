#!/usr/bin/env python3
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>
"""Build both mep-cctv modes and compare them head to head.

  onnx    CCTV_MODE=onnx       the original model, executed by ONNX Runtime
  native  CCTV_MODE=native     TVM model with --real-model, otherwise placeholder

Each mode uses a separate build directory and benchmark subprocess.
Placeholder timings do not measure model inference.

Usage:
  python bench/compare.py --real-model        # compare inference modes
  python bench/compare.py --frames 100        # shorter run
  python bench/compare.py --skip-build        # reuse existing build dirs
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModeSpec:
    """One of the deployable build configurations under comparison."""

    label: str
    generated: bool = False

    BUILD_TYPE: str = "Release"

    @staticmethod
    def default_modes(generated: bool = False) -> List["ModeSpec"]:
        """Return the two deployable modes.

        With generated=True the native mode links the real TVM-compiled model
        instead of the placeholder. Leave it False and native measures only
        pre/post-processing, because the placeholder performs no inference.
        """
        return [
            ModeSpec("onnx"),
            ModeSpec("native", generated),
        ]

    def cmake_args(self) -> List[str]:
        """Return the -D flags that select this mode."""
        args = [
            f"-DCMAKE_BUILD_TYPE={self.BUILD_TYPE}",
            f"-DCCTV_MODE={self.label}",
            "-DBUILD_TESTS=OFF",
        ]
        # Only native mode selects between real and placeholder models.
        if self.label == "native":
            args.append(f"-DCCTV_MODEL={'real' if self.generated else 'placeholder'}")
        return args

    def build_dir(self, root: Path) -> Path:
        """Return this mode's dedicated build directory under root."""
        return root / f"build-{self.label}"


@dataclass(frozen=True)
class ModeMetrics:
    """Measured results for one mode."""

    label: str
    backend: str
    simd: bool
    video_mean_ms: float
    audio_mean_ms: float
    peak_rss_mb: float
    lib_bytes: int


class WinnerPicker:
    """Answers 'which mode wins this metric?'."""

    def best(
        self, results: List[ModeMetrics], key: str, lower_is_better: bool
    ) -> Optional[str]:
        """Return the label of the winning mode, or None if nothing ran."""
        usable = [r for r in results if getattr(r, key)]
        if not usable:
            return None
        chooser = min if lower_is_better else max
        return chooser(usable, key=lambda r: getattr(r, key)).label


class ComparisonReporter:
    """Renders the student-facing comparison table."""

    HEADER: str = (
        "| Mode | Backend | SIMD | Video mean (ms) "
        "| Audio mean (ms) | Peak RSS (MB) | libcctv (KB) |"
    )
    SEPARATOR: str = (
        "|------|---------|------|-----------------"
        "|-----------------|---------------|--------------|"
    )
    BYTES_PER_KB: int = 1024

    def write(self, results: List[ModeMetrics], path: Path) -> None:
        """Write the rendered comparison to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(results))

    def render(self, results: List[ModeMetrics]) -> str:
        """Return the full Markdown report."""
        parts = ["# mep-cctv: three-mode comparison", ""]
        parts.extend(self._table(results))
        parts.extend(self._verdict(results))
        parts.extend(self._reading_guide())
        return "\n".join(parts) + "\n"

    def _table(self, results: List[ModeMetrics]) -> List[str]:
        rows = [self.HEADER, self.SEPARATOR]
        rows.extend(self._row(r) for r in results)
        rows.append("")
        return rows

    def _row(self, r: ModeMetrics) -> str:
        return (
            f"| {r.label} | {r.backend} | {'ON' if r.simd else 'OFF'} "
            f"| {r.video_mean_ms:.2f} "
            f"| {r.audio_mean_ms:.2f} | {r.peak_rss_mb:.1f} "
            f"| {r.lib_bytes / self.BYTES_PER_KB:.1f} |"
        )

    def _verdict(self, results: List[ModeMetrics]) -> List[str]:
        picker = WinnerPicker()
        lines = ["## Verdict", ""]
        lines.append(
            self._claim("Fastest video", picker, results, "video_mean_ms", True)
        )
        lines.append(self._claim("Lowest memory", picker, results, "peak_rss_mb", True))
        lines.append(
            self._claim("Smallest library", picker, results, "lib_bytes", True)
        )
        lines.append("")
        return lines

    def _claim(
        self,
        title: str,
        picker: WinnerPicker,
        results: List[ModeMetrics],
        key: str,
        lower: bool,
    ) -> str:
        winner = picker.best(results, key, lower)
        return f"- **{title}:** {winner if winner else 'no data'}"

    def _reading_guide(self) -> List[str]:
        return [
            "## How to read this",
            "",
            "- Both modes run the same YOLOv8n at the same input size and get",
            "  the whole CPU, so the difference is purely how the model is",
            "  executed: an inference runtime interpreting a graph, versus",
            "  compiled machine code.",
            "- Latency is milliseconds per inference call, measured in a tight",
            "  loop with no camera and no JPEG encoding. It is NOT the frame rate",
            "  the CCTV runs at -- that is capped by the webcam, typically 30/s.",
            "  Divide 1000 by the figure if you want a rate.",
            "- Peak RSS is the whole benchmark process, so it includes the",
            "  Python interpreter -- compare modes against each other, not",
            "  against an absolute budget.",
            "- onnx loads a .onnx file and ONNX Runtime at run time; native",
            "  loads neither, which is why their libcctv sizes differ so much.",
            "- The order of the rows is not a ranking. Read the Verdict for that.",
        ]


class ModeRunner:
    """Builds one mode and benchmarks it in a subprocess."""

    LIB_NAMES: Tuple[str, ...] = ("libcctv.so", "libcctv.dylib")
    BENCH_SCRIPT: str = "benchmark.py"

    def __init__(self, spec: ModeSpec, root: Path) -> None:
        self._spec = spec
        self._root = root

    def build(self) -> None:
        """Configure and compile this mode; raise RuntimeError on failure."""
        build_dir = self._spec.build_dir(self._root)
        self._run(["cmake", "-B", str(build_dir), *self._spec.cmake_args()])
        self._run(["cmake", "--build", str(build_dir), "--parallel"])

    def _run(self, cmd: List[str]) -> None:
        result = subprocess.run(
            cmd, cwd=str(self._root), capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cmake step failed for {self._spec.label}: "
                f"{' '.join(cmd)}\n{result.stderr}"
            )

    def library_path(self) -> Optional[Path]:
        """Return the built libcctv for this mode, if present."""
        build_dir = self._spec.build_dir(self._root)
        for name in self.LIB_NAMES:
            candidate = build_dir / name
            if candidate.exists():
                return candidate
        return None

    def library_size(self, path: Path) -> int:
        """Return the library size in bytes, or 0 when absent."""
        return path.stat().st_size if path.exists() else 0

    def benchmark(self, out_dir: Path, frames: int, warmup: int) -> Optional[Path]:
        """Run benchmark.py against this mode's library; return report path."""
        lib = self.library_path()
        if lib is None:
            return None
        cmd = self._bench_command(lib, out_dir, frames, warmup)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return out_dir / f"{self._spec.label}.json"

    def _bench_command(
        self, lib: Path, out_dir: Path, frames: int, warmup: int
    ) -> List[str]:
        return [
            sys.executable,
            str(Path(__file__).parent / self.BENCH_SCRIPT),
            "--lib",
            str(lib),
            "--label",
            self._spec.label,
            "--frames",
            str(frames),
            "--warmup",
            str(warmup),
            "--out",
            str(out_dir),
            "--format",
            "json",
        ]

    def parse_report(self, path: Path) -> ModeMetrics:
        """Parse a benchmark JSON report into ModeMetrics."""
        data: Dict[str, Any] = json.loads(path.read_text())
        lib = self.library_path()
        return ModeMetrics(
            label=data["label"],
            backend=data["backend"],
            simd=bool(data["simd"]),
            video_mean_ms=float(data["video"]["mean_ms"]),
            audio_mean_ms=float(data["audio"]["mean_ms"]),
            peak_rss_mb=float(data.get("peak_rss_mb", 0.0)),
            lib_bytes=self.library_size(lib) if lib else 0,
        )


class ComparisonDriver:
    """Orchestrates build, benchmark, and reporting for the configured modes."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._root = Path(args.root).resolve()

    def execute(self) -> None:
        """Run every mode and write the comparison report."""
        modes = ModeSpec.default_modes(self._args.real_model)
        results = [self._one_mode(spec) for spec in modes]
        measured = [r for r in results if r is not None]
        reporter = ComparisonReporter()
        dest = Path(self._args.out) / "comparison.md"
        reporter.write(measured, dest)
        print(reporter.render(measured))
        print(f"wrote: {dest}")

    def _one_mode(self, spec: ModeSpec) -> Optional[ModeMetrics]:
        runner = ModeRunner(spec, self._root)
        if not self._prepare(runner, spec):
            return None
        report = runner.benchmark(
            Path(self._args.out), self._args.frames, self._args.warmup
        )
        if report is None or not report.exists():
            print(f"  {spec.label}: benchmark did not produce a report -- skipped")
            return None
        return runner.parse_report(report)

    def _prepare(self, runner: ModeRunner, spec: ModeSpec) -> bool:
        if self._args.skip_build:
            return True
        model = "real" if spec.generated else "placeholder"
        print(f"building {spec.label} (CCTV_MODE={spec.label} model={model}) ...")
        try:
            runner.build()
            return True
        except RuntimeError as exc:
            print(f"  {spec.label}: build failed -- skipped\n    {exc}")
            return False


class CompareCli:
    """Parses arguments for the comparison driver."""

    DEFAULT_OUT: str = str(Path(__file__).parent / "reports")
    DEFAULT_FRAMES: int = 100
    DEFAULT_WARMUP: int = 10

    def build_parser(self) -> argparse.ArgumentParser:
        """Return a configured ArgumentParser."""
        p = argparse.ArgumentParser(description="Compare the three mep-cctv modes")
        p.add_argument("--root", default=str(Path(__file__).parent.parent))
        p.add_argument("--out", default=self.DEFAULT_OUT)
        p.add_argument("--frames", type=int, default=self.DEFAULT_FRAMES)
        p.add_argument("--warmup", type=int, default=self.DEFAULT_WARMUP)
        p.add_argument(
            "--real-model",
            action="store_true",
            help=(
                "Link the real TVM-compiled model into the native modes. "
                "Without this they run the placeholder, which performs no "
                "inference, so their timings are not comparable to mode 1."
            ),
        )
        p.add_argument(
            "--skip-build",
            action="store_true",
            help="Reuse existing build-mode* directories instead of rebuilding.",
        )
        return p

    def execute(self, argv: Optional[List[str]] = None) -> None:
        """Parse argv and run the comparison."""
        ComparisonDriver(self.build_parser().parse_args(argv)).execute()


def main() -> None:
    """CLI entry point."""
    CompareCli().execute()


if __name__ == "__main__":
    main()
