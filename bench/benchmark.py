#!/usr/bin/env python3
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>
"""Benchmark driver for mep-cctv inference engine.

Runs detect_objects and detect_audio over synthetic frames/samples,
collects latency stats, and exports JSON / CSV / Markdown reports.
Multiple labelled runs are aggregated into a single Markdown table.
"""

import argparse
import csv
import json
import os
import resource
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running this file directly (python bench/benchmark.py) as well as
# importing it as part of the bench package.
_BENCH_DIR = Path(__file__).parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from cctv_binding import CctvEngine, CctvLibrary  # noqa: E402
from sources import SyntheticVideoSource  # noqa: E402


@dataclass(frozen=True)
class LatencyStats:
    """Immutable latency statistics in milliseconds, without a derived FPS field."""

    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class RunResult:
    """Full result for one benchmark label (one labelled run)."""

    label: str
    backend: str
    simd: bool
    video: LatencyStats
    audio: LatencyStats
    peak_rss_mb: float = 0.0


class MemoryProbe:
    """Reports the peak resident set size of the current process."""

    # ru_maxrss is bytes on macOS and KiB on Linux.
    BYTES_PER_MB: float = 1024.0 * 1024.0
    KB_PER_MB: float = 1024.0

    def peak_rss_mb(self) -> float:
        """Return peak RSS for this process in megabytes."""
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(usage) / self.BYTES_PER_MB
        return float(usage) / self.KB_PER_MB


class StatsCalculator:
    """Computes latency statistics from a list of millisecond samples."""

    P95_QUANTILE: float = 0.95

    def compute(self, samples_ms: List[float]) -> LatencyStats:
        """Return LatencyStats for the given sample list."""
        if not samples_ms:
            raise ValueError("samples_ms must not be empty")
        mean = statistics.mean(samples_ms)
        median = statistics.median(samples_ms)
        p95 = self._percentile(samples_ms, self.P95_QUANTILE)
        return LatencyStats(
            count=len(samples_ms),
            mean_ms=mean,
            median_ms=median,
            p95_ms=p95,
            min_ms=min(samples_ms),
            max_ms=max(samples_ms),
        )

    def _percentile(self, data: List[float], q: float) -> float:
        sorted_data = sorted(data)
        idx = q * (len(sorted_data) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
        frac = idx - lo
        return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


class VideoTimingRunner:
    """Runs detect_objects over synthetic frames and returns latency samples."""

    FRAME_HEIGHT: int = 720
    FRAME_WIDTH: int = 1280

    def __init__(self, engine: CctvEngine, warmup: int, frames: int) -> None:
        self._engine = engine
        self._warmup = warmup
        self._frames = frames
        self._source = SyntheticVideoSource()

    def run(self) -> List[float]:
        """Return per-inference latency in milliseconds."""
        self._run_warmup()
        return self._collect_samples()

    def _run_warmup(self) -> None:  # pragma: no cover
        for _ in range(self._warmup):
            frame = self._source.read()
            if frame is not None:
                self._engine.detect_objects(frame)

    def _collect_samples(self) -> List[float]:
        samples: List[float] = []
        for _ in range(self._frames):
            frame = self._source.read()
            if frame is None:
                continue
            t0 = time.perf_counter()
            self._engine.detect_objects(frame)
            samples.append((time.perf_counter() - t0) * 1000.0)
        return samples


class AudioTimingRunner:
    """Runs detect_audio over synthetic samples and returns latency samples."""

    SAMPLE_RATE: int = 16000
    CHUNK_SAMPLES: int = 16000
    DEFAULT_RUNS: int = 50

    def __init__(self, engine: CctvEngine, warmup: int, runs: int) -> None:
        self._engine = engine
        self._warmup = warmup
        self._runs = runs
        self._samples = np.zeros(self.CHUNK_SAMPLES, dtype=np.float32)

    def run(self) -> List[float]:
        """Return per-inference latency in milliseconds."""
        self._run_warmup()
        return self._collect_samples()

    def _run_warmup(self) -> None:  # pragma: no cover
        for _ in range(self._warmup):
            self._engine.detect_audio(self._samples, self.SAMPLE_RATE)

    def _collect_samples(self) -> List[float]:
        results: List[float] = []
        for _ in range(self._runs):
            t0 = time.perf_counter()
            self._engine.detect_audio(self._samples, self.SAMPLE_RATE)
            results.append((time.perf_counter() - t0) * 1000.0)
        return results


class JsonReporter:
    """Serialises a RunResult to a JSON file."""

    def write(self, result: RunResult, path: Path) -> None:
        """Write a JSON report file."""
        path.write_text(json.dumps(self._to_dict(result), indent=2))

    def _to_dict(self, result: RunResult) -> Dict[str, Any]:
        return {
            "label": result.label,
            "backend": result.backend,
            "simd": result.simd,
            "peak_rss_mb": round(result.peak_rss_mb, 2),
            "video": self._stats_to_dict(result.video),
            "audio": self._stats_to_dict(result.audio),
        }

    def _stats_to_dict(self, s: LatencyStats) -> Dict[str, Any]:
        return {
            "count": s.count,
            "mean_ms": round(s.mean_ms, 3),
            "median_ms": round(s.median_ms, 3),
            "p95_ms": round(s.p95_ms, 3),
            "min_ms": round(s.min_ms, 3),
            "max_ms": round(s.max_ms, 3),
        }


class CsvReporter:
    """Writes a CSV file with per-inference latency columns."""

    HEADER: List[str] = [
        "label",
        "task",
        "count",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
    ]

    def write(self, result: RunResult, path: Path) -> None:
        """Write a CSV report file."""
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.HEADER)
            writer.writerow(self._row(result, "video", result.video))
            writer.writerow(self._row(result, "audio", result.audio))

    def _row(self, result: RunResult, task: str, s: LatencyStats) -> List[Any]:
        return [
            result.label,
            task,
            s.count,
            round(s.mean_ms, 3),
            round(s.median_ms, 3),
            round(s.p95_ms, 3),
            round(s.min_ms, 3),
            round(s.max_ms, 3),
        ]


class MarkdownReporter:
    """Writes an aligned Markdown summary table from one or more RunResults."""

    COL_WIDTHS: Dict[str, int] = {}

    def write(self, results: List[RunResult], path: Path) -> None:
        """Write a Markdown table aggregating all provided results."""
        path.write_text(self._render(results))

    def _render(self, results: List[RunResult]) -> str:
        rows = [self._header_row()]
        rows.append(self._separator_row())
        for r in results:
            rows.append(self._data_row(r, "video", r.video))
            rows.append(self._data_row(r, "audio", r.audio))
        return "# mep-cctv Inference Benchmark\n\n" + "\n".join(rows) + "\n"

    def _header_row(self) -> str:
        return (
            "| Label | Backend | SIMD | Task "
            "| Mean (ms) | Median (ms) | p95 (ms) "
            "| Min (ms) | Max (ms) |"
        )

    def _separator_row(self) -> str:
        return (
            "|-------|---------|------|------"
            "|-----------|-------------|----------"
            "|----------|----------|"
        )

    def _data_row(self, result: RunResult, task: str, s: LatencyStats) -> str:
        simd = "ON" if result.simd else "OFF"
        return (
            f"| {result.label} | {result.backend} | {simd} | {task} "
            f"| {s.mean_ms:.2f} | {s.median_ms:.2f} | {s.p95_ms:.2f} "
            f"| {s.min_ms:.2f} | {s.max_ms:.2f} |"
        )


class ReportAggregator:
    """Loads existing JSON reports from a directory and merges them."""

    def load_existing(self, out_dir: Path) -> List[RunResult]:
        """Return RunResult objects parsed from all JSON files in out_dir."""
        results: List[RunResult] = []
        for p in sorted(out_dir.glob("*.json")):
            try:
                results.append(self._parse_json(p))
            except (KeyError, json.JSONDecodeError):
                pass
        return results

    def _parse_json(self, path: Path) -> RunResult:
        data = json.loads(path.read_text())
        return RunResult(
            label=data["label"],
            backend=data["backend"],
            simd=bool(data["simd"]),
            video=self._parse_stats(data["video"]),
            audio=self._parse_stats(data["audio"]),
            peak_rss_mb=float(data.get("peak_rss_mb", 0.0)),
        )

    def _parse_stats(self, d: Dict[str, Any]) -> LatencyStats:
        return LatencyStats(
            count=int(d["count"]),
            mean_ms=float(d["mean_ms"]),
            median_ms=float(d["median_ms"]),
            p95_ms=float(d["p95_ms"]),
            min_ms=float(d["min_ms"]),
            max_ms=float(d["max_ms"]),
        )


class ReportWriter:
    """Writes one or more format files given a RunResult and format spec."""

    FORMATS: Tuple[str, ...] = ("json", "csv", "md", "all")

    def __init__(self, out_dir: Path, fmt: str) -> None:
        self._out_dir = out_dir
        self._fmt = fmt
        self._json_reporter = JsonReporter()
        self._csv_reporter = CsvReporter()
        self._md_reporter = MarkdownReporter()
        self._aggregator = ReportAggregator()

    def write(self, result: RunResult) -> List[Path]:
        """Write all requested format files; return paths written."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        if self._fmt in ("json", "all"):
            written.append(self._write_json(result))
        if self._fmt in ("csv", "all"):
            written.append(self._write_csv(result))
        if self._fmt in ("md", "all"):
            written.append(self._write_markdown(result))
        return written

    def _write_json(self, result: RunResult) -> Path:
        p = self._out_dir / f"{result.label}.json"
        self._json_reporter.write(result, p)
        return p

    def _write_csv(self, result: RunResult) -> Path:
        p = self._out_dir / f"{result.label}.csv"
        self._csv_reporter.write(result, p)
        return p

    def _write_markdown(self, result: RunResult) -> Path:
        existing = self._aggregator.load_existing(self._out_dir)
        labels = {r.label for r in existing}
        merged = [r for r in existing if r.label != result.label]
        merged.append(result)
        p = self._out_dir / "summary.md"
        self._md_reporter.write(merged, p)
        return p


class BenchmarkRunner:
    """Orchestrates timing runs and delegates report writing."""

    DEFAULT_FRAMES: int = 300
    DEFAULT_WARMUP: int = 10
    DEFAULT_AUDIO_RUNS: int = 50
    DEFAULT_AUDIO_WARMUP: int = 5

    def __init__(
        self,
        library: CctvLibrary,
        label: str,
        frames: int = DEFAULT_FRAMES,
        warmup: int = DEFAULT_WARMUP,
    ) -> None:
        self._library = library
        self._label = label
        self._frames = frames
        self._warmup = warmup
        self._calculator = StatsCalculator()

    def run(self) -> RunResult:
        """Execute the full benchmark and return a RunResult."""
        with CctvEngine(self._library) as engine:
            backend = engine.backend
            simd = engine.simd_enabled
            video_samples = self._time_video(engine)
            audio_samples = self._time_audio(engine)
        return RunResult(
            label=self._label,
            backend=backend,
            simd=simd,
            video=self._calculator.compute(video_samples),
            audio=self._calculator.compute(audio_samples),
            peak_rss_mb=MemoryProbe().peak_rss_mb(),
        )

    def _time_video(self, engine: CctvEngine) -> List[float]:
        runner = VideoTimingRunner(engine, self._warmup, self._frames)
        return runner.run()

    def _time_audio(self, engine: CctvEngine) -> List[float]:
        runner = AudioTimingRunner(
            engine, self.DEFAULT_AUDIO_WARMUP, self.DEFAULT_AUDIO_RUNS
        )
        return runner.run()


class CliDriver:
    """Parses CLI arguments and drives a benchmark run end-to-end."""

    DEFAULT_OUT: str = str(Path(__file__).parent / "reports")
    VALID_FORMATS: Tuple[str, ...] = ("json", "csv", "md", "all")

    def build_parser(self) -> argparse.ArgumentParser:
        """Return a configured ArgumentParser."""
        p = argparse.ArgumentParser(description="mep-cctv inference benchmark")
        p.add_argument("--lib", default=None, help="Path to libcctv")
        p.add_argument("--label", default="run", help="Label for this run")
        p.add_argument("--frames", type=int, default=BenchmarkRunner.DEFAULT_FRAMES)
        p.add_argument("--warmup", type=int, default=BenchmarkRunner.DEFAULT_WARMUP)
        p.add_argument("--out", default=self.DEFAULT_OUT)
        p.add_argument(
            "--format", dest="fmt", choices=list(self.VALID_FORMATS), default="all"
        )
        return p

    def execute(self, argv: Optional[List[str]] = None) -> None:
        """Parse argv, run benchmark, write reports, print summary."""
        args = self.build_parser().parse_args(argv)
        if args.lib:
            os.environ["CCTV_LIB"] = args.lib
        library = CctvLibrary()
        runner = BenchmarkRunner(library, args.label, args.frames, args.warmup)
        result = runner.run()
        writer = ReportWriter(Path(args.out), args.fmt)
        written = writer.write(result)
        self._print_summary(result, written)

    def _print_summary(self, result: RunResult, written: List[Path]) -> None:
        print(f"label   : {result.label}")
        print(f"backend : {result.backend}  simd={result.simd}")
        print(f"peak RSS: {result.peak_rss_mb:.1f} MB")
        self._print_task_stats("video", result.video)
        self._print_task_stats("audio", result.audio)
        for p in written:
            print(f"wrote   : {p}")

    def _print_task_stats(self, task: str, s: LatencyStats) -> None:
        print(f"{task:<7} : mean={s.mean_ms:.2f} ms  p95={s.p95_ms:.2f} ms")


def main() -> None:
    """CLI entry point."""
    CliDriver().execute()


if __name__ == "__main__":
    main()
