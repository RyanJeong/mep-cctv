# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

"""Tests for bench/benchmark.py -- pure logic, no hardware required."""

import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_BENCH_DIR = Path(__file__).parent.parent
for _d in (_BENCH_DIR,):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from benchmark import (
    AudioTimingRunner,
    BenchmarkRunner,
    CliDriver,
    CsvReporter,
    JsonReporter,
    LatencyStats,
    MarkdownReporter,
    ReportAggregator,
    ReportWriter,
    RunResult,
    StatsCalculator,
    VideoTimingRunner,
)
from cctv_binding import AudioResult, CctvEngine, CctvLibrary, Detection

# ---------------------------------------------------------------------------
# StatsCalculator
# ---------------------------------------------------------------------------


class TestStatsCalculator:
    """Tests for StatsCalculator.compute."""

    def test_mean_of_known_values(self) -> None:
        calc = StatsCalculator()
        stats = calc.compute([10.0, 20.0, 30.0])
        assert stats.mean_ms == pytest.approx(20.0)

    def test_median_odd_count(self) -> None:
        calc = StatsCalculator()
        stats = calc.compute([1.0, 2.0, 100.0])
        assert stats.median_ms == pytest.approx(2.0)

    def test_p95_single_element(self) -> None:
        calc = StatsCalculator()
        stats = calc.compute([42.0])
        assert stats.p95_ms == pytest.approx(42.0)

    def test_p95_two_elements(self) -> None:
        calc = StatsCalculator()
        stats = calc.compute([0.0, 100.0])
        assert stats.p95_ms == pytest.approx(95.0)

    def test_min_max(self) -> None:
        calc = StatsCalculator()
        stats = calc.compute([5.0, 3.0, 8.0, 1.0])
        assert stats.min_ms == pytest.approx(1.0)
        assert stats.max_ms == pytest.approx(8.0)

    def test_count_matches_input(self) -> None:
        calc = StatsCalculator()
        stats = calc.compute([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.count == 5

    def test_empty_raises(self) -> None:
        calc = StatsCalculator()
        with pytest.raises(ValueError, match="empty"):
            calc.compute([])

    def test_p95_five_elements(self) -> None:
        # idx = 0.95 * 4 = 3.8  -> lo=3, hi=4, frac=0.8
        # result = 40 + 0.8 * (50 - 40) = 48.0
        calc = StatsCalculator()
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calc.compute(samples)
        assert stats.p95_ms == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# LatencyStats / RunResult -- dataclass immutability
# ---------------------------------------------------------------------------


class TestLatencyStats:
    """Tests for LatencyStats frozen dataclass."""

    def _make(self) -> LatencyStats:
        return LatencyStats(
            count=10,
            mean_ms=5.0,
            median_ms=4.5,
            p95_ms=9.0,
            min_ms=1.0,
            max_ms=10.0,
        )

    def test_fields_accessible(self) -> None:
        s = self._make()
        assert s.count == 10

    def test_frozen(self) -> None:
        s = self._make()
        with pytest.raises(AttributeError):
            s.mean_ms = 99.0  # type: ignore[misc]


class TestRunResult:
    """Tests for RunResult frozen dataclass."""

    def _make_stats(self) -> LatencyStats:
        return LatencyStats(1, 1.0, 1.0, 1.0, 1.0, 1.0)

    def test_fields_accessible(self) -> None:
        s = self._make_stats()
        r = RunResult("lbl", "native-tvm", True, s, s)
        assert r.label == "lbl"
        assert r.simd is True

    def test_frozen(self) -> None:
        s = self._make_stats()
        r = RunResult("lbl", "native-tvm", True, s, s)
        with pytest.raises(AttributeError):
            r.label = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# JsonReporter
# ---------------------------------------------------------------------------


class TestJsonReporter:
    """Tests for JsonReporter."""

    def _make_result(self) -> RunResult:
        s = LatencyStats(50, 5.0, 4.5, 9.0, 1.0, 12.0)
        return RunResult("test-run", "native-tvm", True, s, s)

    def test_writes_file(self, tmp_path: Path) -> None:
        reporter = JsonReporter()
        result = self._make_result()
        p = tmp_path / "test.json"
        reporter.write(result, p)
        assert p.exists()

    def test_json_has_required_fields(self, tmp_path: Path) -> None:
        reporter = JsonReporter()
        result = self._make_result()
        p = tmp_path / "test.json"
        reporter.write(result, p)
        data = json.loads(p.read_text())
        assert data["label"] == "test-run"
        assert data["backend"] == "native-tvm"
        assert data["simd"] is True
        assert "video" in data
        assert "audio" in data

    def test_json_video_stats_fields(self, tmp_path: Path) -> None:
        reporter = JsonReporter()
        result = self._make_result()
        p = tmp_path / "out.json"
        reporter.write(result, p)
        data = json.loads(p.read_text())
        v = data["video"]
        for key in (
            "count",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
        ):
            assert key in v

    def test_json_values_are_rounded(self, tmp_path: Path) -> None:
        reporter = JsonReporter()
        s = LatencyStats(10, 1.23456789, 1.0, 2.0, 0.5, 5.0)
        r = RunResult("lbl", "native-tvm", False, s, s)
        p = tmp_path / "x.json"
        reporter.write(r, p)
        data = json.loads(p.read_text())
        # mean should be rounded to 3 decimal places
        assert data["video"]["mean_ms"] == pytest.approx(1.235, abs=0.001)


# ---------------------------------------------------------------------------
# CsvReporter
# ---------------------------------------------------------------------------


class TestCsvReporter:
    """Tests for CsvReporter."""

    def _make_result(self) -> RunResult:
        s = LatencyStats(30, 3.0, 2.8, 6.0, 1.0, 8.0)
        return RunResult("csv-run", "native-tvm", False, s, s)

    def test_writes_file(self, tmp_path: Path) -> None:
        reporter = CsvReporter()
        result = self._make_result()
        p = tmp_path / "out.csv"
        reporter.write(result, p)
        assert p.exists()

    def test_csv_has_header(self, tmp_path: Path) -> None:
        reporter = CsvReporter()
        result = self._make_result()
        p = tmp_path / "out.csv"
        reporter.write(result, p)
        lines = p.read_text().splitlines()
        assert "label" in lines[0]
        assert "mean_ms" in lines[0]

    def test_csv_has_two_data_rows(self, tmp_path: Path) -> None:
        reporter = CsvReporter()
        result = self._make_result()
        p = tmp_path / "out.csv"
        reporter.write(result, p)
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 3  # header + video + audio

    def test_csv_label_in_rows(self, tmp_path: Path) -> None:
        reporter = CsvReporter()
        result = self._make_result()
        p = tmp_path / "out.csv"
        reporter.write(result, p)
        content = p.read_text()
        assert "csv-run" in content

    def test_csv_tasks_video_and_audio(self, tmp_path: Path) -> None:
        reporter = CsvReporter()
        result = self._make_result()
        p = tmp_path / "out.csv"
        reporter.write(result, p)
        content = p.read_text()
        assert "video" in content
        assert "audio" in content


# ---------------------------------------------------------------------------
# MarkdownReporter
# ---------------------------------------------------------------------------


class TestMarkdownReporter:
    """Tests for MarkdownReporter."""

    def _make_result(self, label: str = "run1") -> RunResult:
        s = LatencyStats(20, 2.0, 1.9, 4.0, 0.5, 7.0)
        return RunResult(label, "native-tvm", True, s, s)

    def test_writes_file(self, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        p = tmp_path / "summary.md"
        reporter.write([self._make_result()], p)
        assert p.exists()

    def test_md_contains_header(self, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        p = tmp_path / "summary.md"
        reporter.write([self._make_result()], p)
        content = p.read_text()
        assert "# mep-cctv" in content

    def test_md_has_table_separator(self, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        p = tmp_path / "summary.md"
        reporter.write([self._make_result()], p)
        content = p.read_text()
        assert "---" in content

    def test_md_label_present(self, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        p = tmp_path / "summary.md"
        reporter.write([self._make_result("my-run")], p)
        assert "my-run" in p.read_text()

    def test_md_two_labels_both_present(self, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        p = tmp_path / "summary.md"
        r1 = self._make_result("alpha")
        r2 = self._make_result("beta")
        reporter.write([r1, r2], p)
        content = p.read_text()
        assert "alpha" in content
        assert "beta" in content

    def test_md_simd_on_off(self, tmp_path: Path) -> None:
        reporter = MarkdownReporter()
        s = LatencyStats(10, 1.0, 1.0, 1.0, 1.0, 1.0)
        r_on = RunResult("on", "native-tvm", True, s, s)
        r_off = RunResult("off", "native-tvm", False, s, s)
        p = tmp_path / "summary.md"
        reporter.write([r_on, r_off], p)
        content = p.read_text()
        assert "ON" in content
        assert "OFF" in content


# ---------------------------------------------------------------------------
# ReportAggregator
# ---------------------------------------------------------------------------


class TestReportAggregator:
    """Tests for ReportAggregator.load_existing."""

    def _write_json(self, tmp_path: Path, label: str) -> None:
        s = LatencyStats(10, 5.0, 4.5, 9.0, 1.0, 12.0)
        r = RunResult(label, "native-tvm", True, s, s)
        JsonReporter().write(r, tmp_path / f"{label}.json")

    def test_loads_empty_dir(self, tmp_path: Path) -> None:
        agg = ReportAggregator()
        assert agg.load_existing(tmp_path) == []

    def test_loads_single_json(self, tmp_path: Path) -> None:
        self._write_json(tmp_path, "run1")
        agg = ReportAggregator()
        results = agg.load_existing(tmp_path)
        assert len(results) == 1
        assert results[0].label == "run1"

    def test_loads_multiple_json(self, tmp_path: Path) -> None:
        self._write_json(tmp_path, "run1")
        self._write_json(tmp_path, "run2")
        agg = ReportAggregator()
        results = agg.load_existing(tmp_path)
        assert len(results) == 2

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not json")
        agg = ReportAggregator()
        results = agg.load_existing(tmp_path)
        assert results == []

    def test_roundtrip_fields(self, tmp_path: Path) -> None:
        self._write_json(tmp_path, "myrun")
        agg = ReportAggregator()
        r = agg.load_existing(tmp_path)[0]
        assert r.backend == "native-tvm"
        assert r.simd is True
        assert r.video.count == 10


# ---------------------------------------------------------------------------
# ReportWriter
# ---------------------------------------------------------------------------


class TestReportWriter:
    """Tests for ReportWriter -- all formats."""

    def _make_result(self, label: str = "bench") -> RunResult:
        s = LatencyStats(100, 4.0, 3.8, 7.5, 1.2, 10.0)
        return RunResult(label, "native-tvm", True, s, s)

    def test_json_format_writes_json_only(self, tmp_path: Path) -> None:
        writer = ReportWriter(tmp_path, "json")
        written = writer.write(self._make_result())
        assert len(written) == 1
        assert written[0].suffix == ".json"

    def test_csv_format_writes_csv_only(self, tmp_path: Path) -> None:
        writer = ReportWriter(tmp_path, "csv")
        written = writer.write(self._make_result())
        assert len(written) == 1
        assert written[0].suffix == ".csv"

    def test_md_format_writes_md_only(self, tmp_path: Path) -> None:
        writer = ReportWriter(tmp_path, "md")
        written = writer.write(self._make_result())
        assert len(written) == 1
        assert written[0].suffix == ".md"

    def test_all_format_writes_three_files(self, tmp_path: Path) -> None:
        writer = ReportWriter(tmp_path, "all")
        written = writer.write(self._make_result())
        assert len(written) == 3
        suffixes = {p.suffix for p in written}
        assert {".json", ".csv", ".md"} == suffixes

    def test_creates_out_dir(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new" / "subdir"
        writer = ReportWriter(new_dir, "json")
        writer.write(self._make_result())
        assert new_dir.exists()

    def test_md_aggregates_previous_runs(self, tmp_path: Path) -> None:
        # Write first run via json+md
        writer1 = ReportWriter(tmp_path, "all")
        writer1.write(self._make_result("run-a"))
        # Write second run -- md should contain both labels
        writer2 = ReportWriter(tmp_path, "all")
        writer2.write(self._make_result("run-b"))
        md_path = tmp_path / "summary.md"
        content = md_path.read_text()
        assert "run-a" in content
        assert "run-b" in content


# ---------------------------------------------------------------------------
# VideoTimingRunner
# ---------------------------------------------------------------------------


class TestVideoTimingRunner:
    """Tests for VideoTimingRunner._collect_samples (no warmup tested)."""

    def _make_engine_mock(self) -> MagicMock:
        engine = MagicMock()
        engine.detect_objects.return_value = []
        return engine

    def test_returns_correct_count(self) -> None:
        engine = self._make_engine_mock()
        runner = VideoTimingRunner(engine, warmup=0, frames=5)
        # Bypass warmup
        samples = runner._collect_samples()
        assert len(samples) == 5

    def test_samples_are_positive_floats(self) -> None:
        engine = self._make_engine_mock()
        runner = VideoTimingRunner(engine, warmup=0, frames=3)
        samples = runner._collect_samples()
        assert all(s > 0.0 for s in samples)

    def test_engine_called_n_times(self) -> None:
        engine = self._make_engine_mock()
        runner = VideoTimingRunner(engine, warmup=0, frames=7)
        runner._collect_samples()
        assert engine.detect_objects.call_count == 7


# ---------------------------------------------------------------------------
# AudioTimingRunner
# ---------------------------------------------------------------------------


class TestAudioTimingRunner:
    """Tests for AudioTimingRunner._collect_samples."""

    def _make_engine_mock(self) -> MagicMock:
        engine = MagicMock()
        engine.detect_audio.return_value = AudioResult(person_present=False, score=0.0)
        return engine

    def test_returns_correct_count(self) -> None:
        engine = self._make_engine_mock()
        runner = AudioTimingRunner(engine, warmup=0, runs=4)
        samples = runner._collect_samples()
        assert len(samples) == 4

    def test_samples_positive(self) -> None:
        engine = self._make_engine_mock()
        runner = AudioTimingRunner(engine, warmup=0, runs=3)
        samples = runner._collect_samples()
        assert all(s > 0.0 for s in samples)

    def test_engine_called_n_times(self) -> None:
        engine = self._make_engine_mock()
        runner = AudioTimingRunner(engine, warmup=0, runs=6)
        runner._collect_samples()
        assert engine.detect_audio.call_count == 6


# ---------------------------------------------------------------------------
# BenchmarkRunner -- integration with real library
# ---------------------------------------------------------------------------


class TestBenchmarkRunnerIntegration:
    """Tests for BenchmarkRunner against the real libcctv."""

    def test_run_returns_run_result(self, lib_path: str) -> None:
        library = CctvLibrary(lib_path)
        runner = BenchmarkRunner(library, "test", frames=5, warmup=0)
        result = runner.run()
        assert isinstance(result, RunResult)

    def test_label_preserved(self, lib_path: str) -> None:
        library = CctvLibrary(lib_path)
        runner = BenchmarkRunner(library, "my-label", frames=5, warmup=0)
        result = runner.run()
        assert result.label == "my-label"

    def test_backend_is_string(self, lib_path: str) -> None:
        library = CctvLibrary(lib_path)
        runner = BenchmarkRunner(library, "lbl", frames=5, warmup=0)
        result = runner.run()
        assert isinstance(result.backend, str)

    def test_simd_is_bool(self, lib_path: str) -> None:
        library = CctvLibrary(lib_path)
        runner = BenchmarkRunner(library, "lbl", frames=5, warmup=0)
        result = runner.run()
        assert isinstance(result.simd, bool)

    def test_video_stats_count_matches_frames(self, lib_path: str) -> None:
        library = CctvLibrary(lib_path)
        runner = BenchmarkRunner(library, "lbl", frames=8, warmup=0)
        result = runner.run()
        assert result.video.count == 8

    def test_audio_stats_count_matches_default(self, lib_path: str) -> None:
        library = CctvLibrary(lib_path)
        runner = BenchmarkRunner(library, "lbl", frames=5, warmup=0)
        result = runner.run()
        assert result.audio.count == BenchmarkRunner.DEFAULT_AUDIO_RUNS

    def test_build_parser_returns_parser(self) -> None:
        driver = CliDriver()
        parser = driver.build_parser()
        assert parser is not None

    def test_default_label(self) -> None:
        driver = CliDriver()
        args = driver.build_parser().parse_args([])
        assert args.label == "run"

    def test_custom_label(self) -> None:
        driver = CliDriver()
        args = driver.build_parser().parse_args(["--label", "my-run"])
        assert args.label == "my-run"

    def test_default_frames(self) -> None:
        driver = CliDriver()
        args = driver.build_parser().parse_args([])
        assert args.frames == BenchmarkRunner.DEFAULT_FRAMES

    def test_custom_frames(self) -> None:
        driver = CliDriver()
        args = driver.build_parser().parse_args(["--frames", "42"])
        assert args.frames == 42

    def test_default_format_is_all(self) -> None:
        driver = CliDriver()
        args = driver.build_parser().parse_args([])
        assert args.fmt == "all"

    def test_lib_arg_sets_env(self, tmp_path: Path, lib_path: str) -> None:
        import os

        out_dir = tmp_path / "out"
        driver = CliDriver()
        driver.execute(
            [
                "--lib",
                lib_path,
                "--label",
                "cli-test",
                "--frames",
                "3",
                "--warmup",
                "0",
                "--out",
                str(out_dir),
                "--format",
                "json",
            ]
        )
        assert os.environ.get("CCTV_LIB") == lib_path
        assert (out_dir / "cli-test.json").exists()

    def test_execute_writes_report(self, tmp_path: Path, lib_path: str) -> None:
        import os

        os.environ["CCTV_LIB"] = lib_path
        out_dir = tmp_path / "reports"
        driver = CliDriver()
        driver.execute(
            [
                "--label",
                "e2e",
                "--frames",
                "3",
                "--warmup",
                "0",
                "--out",
                str(out_dir),
                "--format",
                "md",
            ]
        )
        assert (out_dir / "summary.md").exists()
