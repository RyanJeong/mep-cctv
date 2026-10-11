#!/usr/bin/env python3
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>
"""ctypes binding for the libcctv shared library."""

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class CctvStatus:
    """Mirror of cctv_status enum."""

    OK: int = 0
    ERR_INVALID_ARG: int = 1
    ERR_BACKEND: int = 2
    ERR_ALLOC: int = 3
    ERR_UNSUPPORTED: int = 4

    STATUS_NAMES = {
        0: "CCTV_OK",
        1: "CCTV_ERR_INVALID_ARG",
        2: "CCTV_ERR_BACKEND",
        3: "CCTV_ERR_ALLOC",
        4: "CCTV_ERR_UNSUPPORTED",
    }

    @classmethod
    def name(cls, code: int) -> str:
        """Return the name for a status code."""
        return cls.STATUS_NAMES.get(code, f"UNKNOWN({code})")


class CctvBox(ctypes.Structure):
    """Mirror of cctv_box struct."""

    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("score", ctypes.c_float),
        ("class_id", ctypes.c_int),
    ]


class CctvDetections(ctypes.Structure):
    """Mirror of cctv_detections struct."""

    _fields_ = [
        ("boxes", ctypes.POINTER(CctvBox)),
        ("count", ctypes.c_int),
    ]


class CctvAudioResult(ctypes.Structure):
    """Mirror of cctv_audio_result struct."""

    _fields_ = [
        ("person_present", ctypes.c_int),
        ("score", ctypes.c_float),
    ]


class CctvConfig(ctypes.Structure):
    """Mirror of cctv_config struct."""

    _fields_ = [
        ("yolo_model_path", ctypes.c_char_p),
        ("conf_threshold", ctypes.c_float),
        ("iou_threshold", ctypes.c_float),
    ]


@dataclass(frozen=True)
class Detection:
    """Immutable Python representation of a single detected box."""

    x: float
    y: float
    w: float
    h: float
    score: float
    class_id: int


@dataclass(frozen=True)
class AudioResult:
    """Immutable Python representation of an audio detection result."""

    person_present: bool
    score: float


class CctvError(Exception):
    """Raised when a libcctv call returns a non-OK status."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"libcctv error: {CctvStatus.name(status)} ({status})")


class CctvLibrary:
    """Locates and loads libcctv, declares argtypes/restypes."""

    LIB_ENV_VAR: str = "CCTV_LIB"
    LIB_NAMES: Tuple[str, ...] = ("libcctv.dylib", "libcctv.so")
    SEARCH_DIRS: Tuple[str, ...] = ("build", "build-verify", "build/Release")
    DEFAULT_CONF_THRESHOLD: float = 0.25
    DEFAULT_IOU_THRESHOLD: float = 0.45
    # ONNX mode loads this file at engine creation; native ignores the path.
    # Audio needs no model in any mode -- it is a plain energy detector.
    YOLO_MODEL_ENV_VAR: str = "CCTV_YOLO_MODEL"

    def __init__(self, lib_path: Optional[str] = None) -> None:
        resolved = lib_path or os.environ.get(self.LIB_ENV_VAR)
        self._lib = ctypes.CDLL(resolved if resolved else self._find_library())
        self._declare_signatures()

    def _find_library(self) -> str:
        project_root = Path(__file__).parent.parent
        for search_dir in self.SEARCH_DIRS:
            for name in self.LIB_NAMES:
                candidate = project_root / search_dir / name
                if candidate.exists():
                    return str(candidate)
        raise FileNotFoundError(
            f"libcctv not found. Set {self.LIB_ENV_VAR} or build the project."
        )

    def _declare_signatures(self) -> None:
        self._declare_meta_functions()
        self._declare_engine_functions()
        self._declare_inference_functions()

    def _declare_meta_functions(self) -> None:
        self._lib.cctv_version.argtypes = []
        self._lib.cctv_version.restype = ctypes.c_char_p
        self._lib.cctv_backend.argtypes = []
        self._lib.cctv_backend.restype = ctypes.c_char_p
        self._lib.cctv_simd_enabled.argtypes = []
        self._lib.cctv_simd_enabled.restype = ctypes.c_int

    def _declare_engine_functions(self) -> None:
        self._lib.cctv_engine_create.argtypes = [ctypes.POINTER(CctvConfig)]
        self._lib.cctv_engine_create.restype = ctypes.c_void_p
        self._lib.cctv_engine_destroy.argtypes = [ctypes.c_void_p]
        self._lib.cctv_engine_destroy.restype = ctypes.c_int

    def _declare_inference_functions(self) -> None:
        self._lib.cctv_detect_objects.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(CctvDetections),
        ]
        self._lib.cctv_detect_objects.restype = ctypes.c_int
        self._lib.cctv_free_detections.argtypes = [ctypes.POINTER(CctvDetections)]
        self._lib.cctv_free_detections.restype = ctypes.c_int
        self._lib.cctv_detect_audio.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(CctvAudioResult),
        ]
        self._lib.cctv_detect_audio.restype = ctypes.c_int

    def version(self) -> str:
        """Return the library version string."""
        result = self._lib.cctv_version()
        return result.decode("utf-8") if result else ""

    def backend(self) -> str:
        """Return the compiled backend name."""
        result = self._lib.cctv_backend()
        return result.decode("utf-8") if result else ""

    def simd_enabled(self) -> bool:
        """Return True if SIMD kernels are compiled in."""
        return bool(self._lib.cctv_simd_enabled())

    def _model_path(self, env_var: str) -> Optional[bytes]:
        """Return the configured model path as bytes, or None when unset."""
        value = os.environ.get(env_var)
        return value.encode("utf-8") if value else None

    def create_engine(
        self,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> int:
        """Create a new engine handle. Caller must destroy it."""
        cfg = CctvConfig(
            yolo_model_path=self._model_path(self.YOLO_MODEL_ENV_VAR),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        handle: Optional[int] = self._lib.cctv_engine_create(ctypes.byref(cfg))
        if handle is None:
            raise CctvError(CctvStatus.ERR_BACKEND)
        return handle

    def destroy_engine(self, handle: Union[int, ctypes.c_void_p]) -> None:
        """Destroy an engine handle."""
        status = self._lib.cctv_engine_destroy(handle)
        if status != CctvStatus.OK:
            raise CctvError(status)

    def detect_objects(
        self, handle: Union[int, ctypes.c_void_p], rgb: NDArray[np.generic]
    ) -> List[Detection]:
        """Run object detection; returns a list of Detection objects."""
        self._validate_rgb(rgb)
        height, width = rgb.shape[:2]
        buf = rgb.astype(np.uint8).flatten()
        c_buf = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        dets = CctvDetections()
        status = self._lib.cctv_detect_objects(
            handle, c_buf, width, height, ctypes.byref(dets)
        )
        if status != CctvStatus.OK:
            raise CctvError(status)
        return self._copy_detections(dets)

    def _validate_rgb(self, rgb: NDArray[np.generic]) -> None:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be HxWx3 array")
        if rgb.dtype != np.uint8:
            raise ValueError("rgb must have dtype uint8")

    def _copy_detections(self, dets: CctvDetections) -> List[Detection]:
        results: List[Detection] = []
        for i in range(dets.count):
            b = dets.boxes[i]
            results.append(
                Detection(
                    x=b.x,
                    y=b.y,
                    w=b.w,
                    h=b.h,
                    score=b.score,
                    class_id=b.class_id,
                )
            )
        self._lib.cctv_free_detections(ctypes.byref(dets))
        return results

    def detect_audio(
        self,
        handle: Union[int, ctypes.c_void_p],
        samples: NDArray[np.generic],
        sample_rate: int,
    ) -> AudioResult:
        """Detect sound activity from mono PCM samples."""
        if samples.ndim != 1:
            raise ValueError("samples must be a 1-D array")
        buf = samples.astype(np.float32)
        c_buf = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out = CctvAudioResult()
        status = self._lib.cctv_detect_audio(
            handle, c_buf, len(buf), sample_rate, ctypes.byref(out)
        )
        if status != CctvStatus.OK:
            raise CctvError(status)
        return AudioResult(
            person_present=bool(out.person_present), score=float(out.score)
        )


class CctvEngine:
    """Context-manager wrapper around a CctvLibrary engine handle."""

    def __init__(
        self,
        library: Optional[CctvLibrary] = None,
        conf_threshold: float = CctvLibrary.DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = CctvLibrary.DEFAULT_IOU_THRESHOLD,
    ) -> None:
        self._lib = library if library is not None else CctvLibrary()
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._handle: Optional[int] = None

    def __enter__(self) -> "CctvEngine":
        self._handle = self._lib.create_engine(
            self._conf_threshold, self._iou_threshold
        )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._handle is not None:
            self._lib.destroy_engine(self._handle)
            self._handle = None

    def _require_handle(self) -> int:
        if self._handle is None:
            raise RuntimeError("Engine not started; use as context manager.")
        return self._handle

    def detect_objects(self, rgb: NDArray[np.generic]) -> List[Detection]:
        """Detect objects in an RGB HxWx3 uint8 frame."""
        return self._lib.detect_objects(self._require_handle(), rgb)

    def detect_audio(
        self, samples: NDArray[np.generic], sample_rate: int
    ) -> AudioResult:
        """Detect sound activity in mono float32 samples."""
        return self._lib.detect_audio(self._require_handle(), samples, sample_rate)

    @property
    def version(self) -> str:
        """Return library version string."""
        return self._lib.version()

    @property
    def backend(self) -> str:
        """Return compiled backend name."""
        return self._lib.backend()

    @property
    def simd_enabled(self) -> bool:
        """Return True if SIMD is compiled in."""
        return self._lib.simd_enabled()


def main() -> None:
    """Smoke-test the binding against the built library."""
    lib = CctvLibrary()
    print(f"version : {lib.version()}")
    print(f"backend : {lib.backend()}")
    print(f"simd    : {lib.simd_enabled()}")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with CctvEngine(lib) as engine:
        dets = engine.detect_objects(frame)
        print(f"detections : {dets}")
        audio = engine.detect_audio(np.zeros(16000, dtype=np.float32), 16000)
        print(f"audio      : {audio}")


if __name__ == "__main__":
    main()
