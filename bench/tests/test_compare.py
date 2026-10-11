# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

"""Tests for bench/compare.py -- the inference-mode comparison driver.

Pure logic only: every test that would otherwise shell out to cmake/ctest or
load a real libcctv patches the boundary instead, so the suite stays
deterministic and fast on any machine.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_BENCH_DIR = Path(__file__).parent.parent
for _d in (_BENCH_DIR,):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from compare import (  # noqa: E402
    ComparisonReporter,
    ModeMetrics,
    ModeRunner,
    ModeSpec,
    WinnerPicker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics(
    label: str,
    ms: float = 100.0,
    rss: float = 100.0,
    lib: int = 1000,
    audio: float = 5.0,
) -> ModeMetrics:
    """Build a ModeMetrics with sensible defaults for table/winner tests."""
    return ModeMetrics(
        label=label,
        backend="native",
        simd=True,
        video_mean_ms=ms,
        audio_mean_ms=audio,
        peak_rss_mb=rss,
        lib_bytes=lib,
    )


# ---------------------------------------------------------------------------
# ModeSpec
# ---------------------------------------------------------------------------


class TestModeSpec:
    """ModeSpec describes a build configuration."""

    def test_two_default_modes_are_defined(self) -> None:
        assert len(ModeSpec.default_modes()) == 2

    def test_default_modes_have_unique_labels(self) -> None:
        labels = [m.label for m in ModeSpec.default_modes()]
        assert len(set(labels)) == 2

    def test_first_mode_is_onnx(self) -> None:
        assert ModeSpec.default_modes()[0].label == "onnx"

    def test_modes_are_named_for_what_they_are(self) -> None:
        # Names describe how the model is executed, not a ranking.
        assert [m.label for m in ModeSpec.default_modes()] == ["onnx", "native"]

    def test_onnx_mode_passes_no_model_flag(self) -> None:
        # ONNX mode does not need a native-model selection flag.
        args = ModeSpec.default_modes()[0].cmake_args()
        assert not any("CCTV_MODEL=" in a for a in args)

    def test_native_mode_is_present_exactly_once(self) -> None:
        native = [m for m in ModeSpec.default_modes() if m.label == "native"]
        assert len(native) == 1

    def test_cmake_args_select_the_mode(self) -> None:
        assert "-DCCTV_MODE=native" in ModeSpec("native").cmake_args()
        assert "-DCCTV_MODE=onnx" in ModeSpec("onnx").cmake_args()

    def test_native_requests_the_real_model_when_asked(self) -> None:
        assert "-DCCTV_MODEL=real" in ModeSpec("native", True).cmake_args()

    def test_native_requests_the_placeholder_otherwise(self) -> None:
        assert "-DCCTV_MODEL=placeholder" in ModeSpec("native").cmake_args()

    def test_cmake_args_request_release_build(self) -> None:
        # A benchmark of a Debug build would be meaningless.
        spec = ModeSpec("native")
        assert "-DCMAKE_BUILD_TYPE=Release" in spec.cmake_args()

    def test_build_dir_is_derived_from_label(self) -> None:
        spec = ModeSpec("native")
        assert "native" in str(spec.build_dir(Path("/tmp")))


# ---------------------------------------------------------------------------
# WinnerPicker
# ---------------------------------------------------------------------------


class TestWinnerPicker:
    """WinnerPicker answers 'which mode wins this metric?'."""

    def test_lowest_latency_wins_speed(self) -> None:
        picker = WinnerPicker()
        results = [_metrics("slow", ms=200.0), _metrics("fast", ms=20.0)]
        assert picker.best(results, "video_mean_ms", lower_is_better=True) == "fast"

    def test_lowest_rss_wins_memory(self) -> None:
        picker = WinnerPicker()
        results = [_metrics("fat", rss=900.0), _metrics("lean", rss=40.0)]
        assert picker.best(results, "peak_rss_mb", lower_is_better=True) == "lean"

    def test_smallest_library_wins_size(self) -> None:
        picker = WinnerPicker()
        results = [_metrics("big", lib=9_000_000), _metrics("small", lib=200_000)]
        assert picker.best(results, "lib_bytes", lower_is_better=True) == "small"

    def test_empty_input_returns_none(self) -> None:
        assert WinnerPicker().best([], "video_mean_ms", lower_is_better=True) is None

    def test_single_entry_wins_by_default(self) -> None:
        picker = WinnerPicker()
        assert picker.best([_metrics("only")], "video_mean_ms", True) == "only"

    def test_ignores_entries_with_zero_metric(self) -> None:
        # A mode that failed to run reports 0.0; it must not win "lowest memory".
        picker = WinnerPicker()
        results = [_metrics("failed", rss=0.0), _metrics("ok", rss=50.0)]
        assert picker.best(results, "peak_rss_mb", lower_is_better=True) == "ok"


# ---------------------------------------------------------------------------
# ComparisonReporter
# ---------------------------------------------------------------------------


class TestComparisonReporter:
    """ComparisonReporter renders the student-facing summary."""

    def test_render_contains_every_label(self) -> None:
        out = ComparisonReporter().render([_metrics("a"), _metrics("b")])
        assert "a" in out and "b" in out

    def test_render_marks_the_speed_winner(self) -> None:
        out = ComparisonReporter().render(
            [_metrics("slow", ms=900.0), _metrics("quick", ms=9.0)]
        )
        assert "quick" in out
        assert "fastest" in out.lower()

    def test_render_has_no_fps_column(self) -> None:
        # FPS is just 1000/mean_ms; two columns of the same fact can disagree.
        out = ComparisonReporter().render([_metrics("a")])
        assert "FPS" not in out

    def test_render_reports_lowest_memory(self) -> None:
        out = ComparisonReporter().render(
            [_metrics("lean", rss=10.0), _metrics("fat", rss=800.0)]
        )
        assert "memory" in out.lower()

    def test_render_is_valid_markdown_table(self) -> None:
        out = ComparisonReporter().render([_metrics("a")])
        assert "|" in out and "---" in out

    def test_render_handles_empty_results(self) -> None:
        out = ComparisonReporter().render([])
        assert isinstance(out, str)

    def test_render_explains_what_differs_between_modes(self) -> None:
        out = ComparisonReporter().render(ModeRunnerStubResults.three())
        assert "compiled machine code" in out

    def test_write_creates_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "summary.md"
        ComparisonReporter().write([_metrics("a")], dest)
        assert dest.exists()

    def test_write_content_matches_render(self, tmp_path: Path) -> None:
        dest = tmp_path / "summary.md"
        results = [_metrics("a")]
        ComparisonReporter().write(results, dest)
        assert dest.read_text() == ComparisonReporter().render(results)


class ModeRunnerStubResults:
    """Canned three-mode result sets used by table tests."""

    @staticmethod
    def three() -> list:
        """Return one ModeMetrics per mode, in mode order."""
        return [
            ModeMetrics("onnx", "dl", False, 100.0, 8.0, 300.0, 900),
            ModeMetrics("native", "native", True, 25.0, 3.0, 118.0, 850),
        ]


# ---------------------------------------------------------------------------
# ModeRunner -- subprocess boundary is mocked
# ---------------------------------------------------------------------------


class TestModeRunner:
    """ModeRunner builds a mode and parses the benchmark JSON it produces."""

    def _report(self, tmp_path: Path, label: str) -> Path:
        payload = {
            "label": label,
            "backend": "native",
            "simd": True,
            "peak_rss_mb": 42.5,
            "video": {"mean_ms": 20.0},
            "audio": {"mean_ms": 3.0},
        }
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload))
        return path

    def test_parse_report_reads_latency(self, tmp_path: Path) -> None:
        path = self._report(tmp_path, "m")
        runner = ModeRunner(ModeSpec("native"), tmp_path)
        assert runner.parse_report(path).video_mean_ms == pytest.approx(20.0)

    def test_parse_report_reads_peak_rss(self, tmp_path: Path) -> None:
        path = self._report(tmp_path, "m")
        runner = ModeRunner(ModeSpec("native"), tmp_path)
        assert runner.parse_report(path).peak_rss_mb == pytest.approx(42.5)

    def test_parse_report_reads_audio_latency(self, tmp_path: Path) -> None:
        path = self._report(tmp_path, "m")
        runner = ModeRunner(ModeSpec("native"), tmp_path)
        assert runner.parse_report(path).audio_mean_ms == pytest.approx(3.0)

    def test_parse_report_tolerates_missing_memory_field(self, tmp_path: Path) -> None:
        path = tmp_path / "x.json"
        path.write_text(
            json.dumps(
                {
                    "label": "x",
                    "backend": "dl",
                    "simd": False,
                    "video": {"mean_ms": 1.0},
                    "audio": {"mean_ms": 1.0},
                }
            )
        )
        runner = ModeRunner(ModeSpec("onnx"), tmp_path)
        assert runner.parse_report(path).peak_rss_mb == 0.0

    def test_library_size_of_missing_file_is_zero(self, tmp_path: Path) -> None:
        runner = ModeRunner(ModeSpec("native"), tmp_path)
        assert runner.library_size(tmp_path / "nope.so") == 0

    def test_library_size_reads_real_file(self, tmp_path: Path) -> None:
        lib = tmp_path / "libcctv.so"
        lib.write_bytes(b"x" * 1234)
        runner = ModeRunner(ModeSpec("native"), tmp_path)
        assert runner.library_size(lib) == 1234

    def test_build_invokes_cmake_configure_and_build(self, tmp_path: Path) -> None:
        runner = ModeRunner(ModeSpec("native"), tmp_path)
        with patch("compare.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0)
            runner.build()
        commands = [call.args[0][0] for call in run.call_args_list]
        assert commands.count("cmake") == 2

    def test_build_raises_on_configure_failure(self, tmp_path: Path) -> None:
        runner = ModeRunner(ModeSpec("native"), tmp_path)
        with patch("compare.subprocess.run") as run:
            run.return_value = MagicMock(returncode=1, stderr="boom")
            with pytest.raises(RuntimeError, match="cmake"):
                runner.build()
