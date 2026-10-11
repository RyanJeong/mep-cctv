# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

"""Shared fixtures for bench tests."""

import sys
from pathlib import Path

import pytest

# Ensure bench/ is importable
_BENCH_DIR = Path(__file__).parent.parent
for d in (_BENCH_DIR,):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


@pytest.fixture(scope="session")
def lib_path() -> str:
    """Return the path to the built libcctv shared library.

    Only the shared-library extension for this platform is considered.
    """
    root = Path(__file__).parent.parent.parent
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    for directory in sorted(root.glob("build*")):
        candidate = directory / f"libcctv{suffix}"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"no libcctv{suffix} under {root}/build*; run a cmake build first."
    )
