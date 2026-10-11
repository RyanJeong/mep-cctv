// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

#ifndef APP_ANNOTATOR_HPP_
#define APP_ANNOTATOR_HPP_

/**
 * @file annotator.hpp
 * @brief Draws detection rectangles into an RGB frame.
 *
 * Boxes are burned into the pixels server-side rather than sent to the
 * browser as coordinates. That is what lets the page be a plain <img> tag
 * fed by an MJPEG stream: the squares are simply part of the picture, so no
 * JavaScript overlay and no WebSocket are needed to keep them aligned with
 * the frame they belong to.
 */

#include <vector>

#include "app/frame.hpp"
#include "cctv/cctv.h"

namespace mep_cctv {
namespace app {

/** @brief Burns bounding boxes into frames. */
class Annotator {
 public:
  /** @brief Rectangle edge thickness in pixels. */
  static const int kBorderThickness;

  /** @brief Number of distinct rectangle colours before the palette repeats. */
  static const int kPaletteSize;

  /**
   * @brief Draw every box onto the frame, each with a colour and a label.
   *
   * Coordinates are clipped to the frame, so a box that extends past an edge
   * is drawn partially rather than corrupting memory.
   *
   * Colour comes from the box's position in @p boxes, not from its class: two
   * people standing together are the common case for a CCTV, and giving them
   * one colour per class would put them back to being indistinguishable. The
   * label carries the class instead. Detections arrive sorted by score, so the
   * assignment is stable frame to frame for a still scene.
   *
   * @param frame  Frame to draw into; must be valid.
   * @param boxes  Detections in original-image pixel space.
   * @param count  Number of entries in @p boxes.
   * @return Number of boxes that were at least partly visible.
   *
   * @side_effects VIOLATION: Non-referentially transparent
   * @side_effects_reason Mutates the caller's frame in place.
   * @side_effects_what   Writes pixels into frame->rgb.
   * @side_effects_impact Bounded by perimeter * thickness per box.
   * @side_effects_alternatives Returning an annotated copy would duplicate a
   *                      full frame on every tick, which at 720p is ~2.7 MB
   *                      of copying per frame for a few hundred edge pixels.
   */
  int Draw(Frame* frame, const cctv_box* boxes, int count) const;

 private:
  /** @brief Draw one rectangle and its label; 1 when any pixel was visible. */
  int DrawBox(Frame* frame, const cctv_box& box, int index) const;

  /** @brief Set one pixel if it lies inside the frame; returns bytes written.
   */
  int SetPixel(Frame* frame, int x, int y, const uint8_t rgb[3]) const;

  /** @brief Fill a rectangle, clipped to the frame; returns pixels written. */
  int FillRect(Frame* frame, int x0, int y0, int x1, int y1,
               const uint8_t rgb[3]) const;

  /**
   * @brief Draw a label band with the class name and score above the box.
   *
   * @param frame Frame to draw into.
   * @param box   Detection the label belongs to.
   * @param x0    Left edge of the rectangle, already clipped.
   * @param y0    Top edge of the rectangle, already clipped.
   * @param rgb   Band colour; the text is drawn in a contrasting shade.
   * @return Pixels written.
   */
  int DrawLabel(Frame* frame, const cctv_box& box, int x0, int y0,
                const uint8_t rgb[3]) const;
};

}  // namespace app
}  // namespace mep_cctv

#endif  // APP_ANNOTATOR_HPP_
