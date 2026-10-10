# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

"""Regression checks for workflow structure and generated-file exclusions."""

import fnmatch
import subprocess
import unittest
from pathlib import Path
from typing import Any, Dict, List

import yaml


class UniqueKeyLoader(yaml.BaseLoader):
    """Reject duplicate YAML mapping keys instead of silently overwriting them."""

    def construct_mapping(self, node: Any, deep: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class TestRepositoryConfig(unittest.TestCase):
    ROOT: Path = Path(__file__).resolve().parents[2]
    EXPECTED_MATRIX = {("onnx", "OFF"), ("native", "OFF"), ("native", "ON")}
    COVERAGE_MINIMUM: float = 80.0

    def workflow(self, name: str) -> Dict[str, Any]:
        path = self.ROOT / ".github" / "workflows" / name
        return yaml.load(path.read_text(), Loader=UniqueKeyLoader)

    def steps(self, workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            step for job in workflow["jobs"].values() for step in job.get("steps", [])
        ]

    def test_all_workflows_have_unique_keys(self) -> None:
        for path in (self.ROOT / ".github" / "workflows").glob("*.yml"):
            with self.subTest(path=path.name):
                self.assertIn("jobs", self.workflow(path.name))

    def test_build_matrix_uses_current_mode_options(self) -> None:
        job = self.workflow("ci.yml")["jobs"]["build-test"]
        rows = job["strategy"]["matrix"]["include"]
        actual = {(row["mode"], row["scalar_kernels"]) for row in rows}
        self.assertEqual(self.EXPECTED_MATRIX, actual)

    def test_native_ci_uses_placeholder_explicitly(self) -> None:
        job = self.workflow("ci.yml")["jobs"]["build-test"]
        configure = next(s["run"] for s in job["steps"] if s.get("name") == "Configure")
        self.assertIn("-DCCTV_MODEL=placeholder", configure)

    def test_learn_ci_is_manual_only(self) -> None:
        workflow = self.workflow("learn-ci.yml")
        self.assertEqual({"workflow_dispatch"}, set(workflow["on"]))

    def test_release_requires_successful_main_push_ci(self) -> None:
        workflow = self.workflow("release.yml")
        self.assertEqual(["CI"], workflow["on"]["workflow_run"]["workflows"])
        self.assertEqual(["completed"], workflow["on"]["workflow_run"]["types"])
        self.assertEqual(["main"], workflow["on"]["workflow_run"]["branches"])
        condition = workflow["jobs"]["build-release"]["if"]
        self.assertIn("conclusion == 'success'", condition)
        self.assertIn("event == 'push'", condition)
        self.assertIn("head_repository.full_name == github.repository", condition)

    def test_release_build_precedes_publication(self) -> None:
        workflow = self.workflow("release.yml")
        build = workflow["jobs"]["build-release"]
        publish = workflow["jobs"]["github-release"]
        self.assertEqual("build-release", publish["needs"])
        self.assertTrue(any(s.get("name") == "Test" for s in build["steps"]))
        for step in self.steps(workflow):
            self.assertNotIn("git push", step.get("run", ""))
            self.assertNotIn("tag_created", str(step))

    def test_release_build_is_a_library_only_placeholder(self) -> None:
        steps = self.workflow("release.yml")["jobs"]["build-release"]["steps"]
        configure = next(s["run"] for s in steps if s.get("name") == "Configure")
        for option in ("CCTV_MODEL=placeholder", "BUILD_APP=OFF", "BUILD_TESTS=ON"):
            self.assertIn(f"-D{option}", configure)
        self.assertNotIn("\\\n\n", configure)

    def test_release_jobs_checkout_the_verified_commit(self) -> None:
        workflow = self.workflow("release.yml")
        for step in self.steps(workflow):
            if step.get("uses", "").startswith("actions/checkout@"):
                self.assertEqual(
                    "${{ github.event.workflow_run.head_sha }}", step["with"]["ref"]
                )

    def test_shell_lint_covers_all_owned_script_directories(self) -> None:
        steps = self.workflow("ci.yml")["jobs"]["lint"]["steps"]
        script = next(
            s["run"] for s in steps if s.get("name") == "Shell script syntax check"
        )
        self.assertIn("scripts models bench", script)
        self.assertIn('bash -n "$script"', script)
        self.assertNotIn("-exec bash -n", script)

    def test_coverage_measures_branches_without_lowering_line_gate(self) -> None:
        steps = self.workflow("ci.yml")["jobs"]["coverage"]["steps"]
        script = "\n".join(s.get("run", "") for s in steps)
        self.assertIn("--branch-coverage", script)
        gate = next(s for s in steps if s.get("name") == "Enforce coverage gate")
        self.assertGreaterEqual(float(gate["env"]["MIN_LINE_COVERAGE"]), 90.0)
        self.assertGreater(
            float(gate["env"]["MIN_BRANCH_COVERAGE"]), self.COVERAGE_MINIMUM
        )

    def test_coverage_merges_llvm_profiles_from_each_test(self) -> None:
        steps = self.workflow("ci.yml")["jobs"]["coverage"]["steps"]
        script = "\n".join(s.get("run", "") for s in steps)
        self.assertIn("-DCMAKE_CXX_COMPILER=clang++", script)
        self.assertIn("llvm-profdata merge", script)
        self.assertIn("llvm-cov export", script)
        test = next(s for s in steps if s.get("name") == "Test")
        self.assertIn("%p-%m.profraw", test["env"]["LLVM_PROFILE_FILE"])

    def test_sanitizers_are_separate_jobs(self) -> None:
        matrix = self.workflow("ci.yml")["jobs"]["sanitizers"]["strategy"]["matrix"]
        self.assertEqual({"ENABLE_ASAN", "ENABLE_TSAN"}, set(matrix["sanitizer"]))

    def test_tsan_runner_prepares_compatible_address_layout(self) -> None:
        steps = self.workflow("ci.yml")["jobs"]["sanitizers"]["steps"]
        setup = next(s for s in steps if s.get("name") == "Prepare TSan address layout")
        self.assertEqual("matrix.sanitizer == 'ENABLE_TSAN'", setup["if"])
        self.assertIn("sudo sysctl -w vm.mmap_rnd_bits=28", setup["run"])

    def test_python_builds_placeholder_before_integration_tests(self) -> None:
        steps = self.workflow("ci.yml")["jobs"]["python"]["steps"]
        names = [step.get("name") for step in steps]
        build = names.index("Build integration library")
        self.assertLess(build, names.index("Tests and coverage"))
        script = steps[build]["run"]
        for option in ("CCTV_MODE=native", "CCTV_MODEL=placeholder", "BUILD_APP=OFF"):
            self.assertIn(f"-D{option}", script)
        self.assertIn("cmake --build build", script)

    def test_python_coverage_excludes_test_code(self) -> None:
        steps = self.workflow("ci.yml")["jobs"]["python"]["steps"]
        script = next(s["run"] for s in steps if s.get("name") == "Tests and coverage")
        self.assertIn("--cov-config=bench/pyproject.toml", script)
        config = (self.ROOT / "bench" / "pyproject.toml").read_text()
        self.assertIn('[tool.coverage.run]\nomit = ["bench/tests/*"]', config)

    def test_docker_context_keeps_required_generated_object(self) -> None:
        relative = "src/backend_native/generated/yolov8n_tuned_kernels.o"
        ignored = False
        for line in (self.ROOT / ".dockerignore").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            pattern = line.lstrip("!").lstrip("/")
            if fnmatch.fnmatch(relative, pattern):
                ignored = not negated
        self.assertFalse(ignored, "real native builds need the generated object")

    def ignore_result(self, paths: List[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "check-ignore", "--no-index", *paths],
            cwd=self.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_ignore_rules_exclude_generated_files(self) -> None:
        paths = ["s-nat/cctv_tests", "coverage-python.xml", "libcctv.so.1"]
        result = self.ignore_result(paths)
        self.assertEqual(paths, result.stdout.splitlines())

    def test_ignore_rules_preserve_sources(self) -> None:
        sources = [
            "models/tuning/rpi4/database_workload.json",
            "tests/fixtures/onnx/yolo_test.onnx",
            ".github/workflows/ci.yml",
        ]
        result = self.ignore_result(sources)
        self.assertEqual(1, result.returncode)
        self.assertEqual("", result.stdout)
