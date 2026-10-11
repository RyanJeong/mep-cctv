// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#include "app/coco_labels.hpp"

#include <array>

namespace mep_cctv {
namespace app {
namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// The 80 COCO classes in the order Ultralytics exports them. The order is
// part of the model's output contract: row 4 + class_id of the detection
// tensor is this class's score, so reordering this table silently mislabels
// every detection.
const std::array<const char*, 80> kNames = {{
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush",
}};

const int kNameCount = static_cast<int>(kNames.size());

}  // namespace

const int kCocoClassCount = kNameCount;

const char* CocoLabel(int class_id) {
  // A corrupt or future model could report anything; returning a string rather
  // than indexing blindly keeps a bad class_id from reading out of bounds.
  if (class_id < 0 || class_id >= kNameCount) return "unknown";

  return kNames[class_id];
}

}  // namespace app
}  // namespace mep_cctv
