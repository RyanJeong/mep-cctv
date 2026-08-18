#!/usr/bin/env python3
# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>
"""Generate the synthetic ONNX test model used by the onnx-backend tests.

Creates tests/fixtures/onnx/yolo_test.onnx:
  Input:  images  float[1,3,S,S]
  Output: output0 float[1,84,21*(S/32)^2]

S must match kInputSize in src/core/image_ops.hpp (currently 320).

The model is a constant output that always returns two high-confidence
detections, at anchor indices 100 and 200. That is enough to exercise the
decode and NMS path without shipping real weights in the repository.

This fixture is used only by GoogleTest; it does not represent real weights.
"""

import os
import struct

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(model: onnx.ModelProto, path: str) -> None:
    """Check and save an ONNX model."""
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(model.SerializeToString())
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# YOLO test model
# ---------------------------------------------------------------------------

def make_yolo_model(output_path: str) -> None:
    """Create a minimal YOLOv8-like ONNX model for testing."""
    # Build the constant output tensor, all zeros except
    # two synthetic detections at anchor indices 100 and 200.
    # Must match src/core/image_ops.hpp kInputSize and the anchor count
    # derived from it in yolo_postprocess.hpp: 21 * (S/32)^2.
    IMG_SIZE: int = 320
    NUM_ANCHORS: int = 21 * (IMG_SIZE // 32) * (IMG_SIZE // 32)
    NUM_ROWS: int = 84  # 4 box coords + 80 class scores

    raw = np.zeros((1, NUM_ROWS, NUM_ANCHORS), dtype=np.float32)

    # Coordinates are fractions of the letterbox side, not absolute pixels,
    # so the synthetic boxes stay inside the frame whatever IMG_SIZE is.
    # Anchor 100: class 0, score 0.92, centred left-of-middle.
    raw[0, 0, 100] = 0.3125 * IMG_SIZE  # cx (letterbox space)
    raw[0, 1, 100] = 0.3125 * IMG_SIZE  # cy
    raw[0, 2, 100] = 0.1563 * IMG_SIZE  # w
    raw[0, 3, 100] = 0.1563 * IMG_SIZE  # h
    raw[0, 4, 100] = 0.92               # class 0 score

    # Anchor 200: class 0, score 0.85, centred right-of-middle.
    raw[0, 0, 200] = 0.6250 * IMG_SIZE
    raw[0, 1, 200] = 0.4688 * IMG_SIZE
    raw[0, 2, 200] = 0.1250 * IMG_SIZE
    raw[0, 3, 200] = 0.1875 * IMG_SIZE
    raw[0, 4, 200] = 0.85

    const_tensor = numpy_helper.from_array(raw, name="output0_const")

    # A single Constant node produces output0.
    node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=const_tensor,
    )

    graph = helper.make_graph(
        [node],
        "yolo_test",
        inputs=[
            helper.make_tensor_value_info(
                "images", onnx.TensorProto.FLOAT, [1, 3, IMG_SIZE, IMG_SIZE]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output0", onnx.TensorProto.FLOAT, [1, NUM_ROWS, NUM_ANCHORS]
            )
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    _save(model, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    fixtures_dir = os.path.join(repo_root, "tests", "fixtures", "onnx")

    make_yolo_model(os.path.join(fixtures_dir, "yolo_test.onnx"))

    print("Done.")


if __name__ == "__main__":
    main()
