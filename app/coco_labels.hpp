// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_COCO_LABELS_HPP_
#define APP_COCO_LABELS_HPP_

/**
 * @file coco_labels.hpp
 * @brief Human-readable names for the COCO class indices YOLOv8n emits.
 *
 * The detector reports a class as an integer, which is all the model knows.
 * Turning that into a word is a presentation concern, so it lives here in the
 * app rather than in the inference library.
 */

namespace mep_cctv {
namespace app {

/** @brief Number of classes in the COCO detection set YOLOv8n was trained on.
 */
extern const int kCocoClassCount;

/**
 * @brief Name for a COCO class index.
 *
 * @param class_id Index as reported in cctv_box::class_id.
 * @return The class name, or "unknown" when the index is out of range.
 *         Never null, so callers can print the result unconditionally.
 */
const char* CocoLabel(int class_id);

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_COCO_LABELS_HPP_
