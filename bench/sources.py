#!/usr/bin/env python3
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>
"""Synthetic frame source for the benchmark harness.

Live camera and microphone capture now live in the C++ application
(app/sources.hpp, app/video_capture_v4l2.cc). What remains here is only what
the benchmark needs: a deterministic frame generator, so timings are
reproducible and never depend on what a camera happened to see.
"""

import numpy as np
from numpy.typing import NDArray


class SyntheticVideoSource:
    """Generates deterministic 1280x720 frames with a moving rectangle."""

    WIDTH: int = 1280
    HEIGHT: int = 720
    RECT_W: int = 100
    RECT_H: int = 80
    SPEED: int = 10
    BG_COLOR: int = 40
    RECT_COLOR: int = 200
    RECT_TOP: int = 100

    def __init__(self) -> None:
        self._frame_idx: int = 0

    def read(self) -> NDArray[np.uint8]:
        """Return the next synthetic RGB frame."""
        frame = np.full((self.HEIGHT, self.WIDTH, 3), self.BG_COLOR, dtype=np.uint8)
        x = self._rect_x()
        frame[self.RECT_TOP : self.RECT_TOP + self.RECT_H, x : x + self.RECT_W] = (
            self.RECT_COLOR
        )
        self._frame_idx += 1
        return frame

    def release(self) -> None:
        """Release resources; nothing to do for a generated source."""

    def _rect_x(self) -> int:
        max_x = self.WIDTH - self.RECT_W
        phase = (self._frame_idx * self.SPEED) % (2 * max_x)
        pos = phase if phase < max_x else 2 * max_x - phase
        return int(pos)
