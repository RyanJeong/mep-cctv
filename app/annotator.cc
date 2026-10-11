// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file annotator.cc
 * @brief Annotator implementation.
 */

#include "app/annotator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

#include "app/coco_labels.hpp"

namespace mep_cctv {
namespace app {

const int Annotator::kBorderThickness = 3;

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

using Rgb = std::array<uint8_t, 3>;
using Glyph = std::array<uint8_t, 5>;

// Bright rectangle colors cycle per detection, starting with green.
const std::array<Rgb, 8> kPalette = {{
    {{0u, 255u, 0u}},     // green
    {{255u, 64u, 64u}},   // red
    {{64u, 160u, 255u}},  // blue
    {{255u, 224u, 0u}},   // yellow
    {{255u, 128u, 0u}},   // orange
    {{224u, 96u, 255u}},  // violet
    {{0u, 224u, 224u}},   // cyan
    {{255u, 255u, 255u}}  // white
}};
const int kPaletteCount = static_cast<int>(kPalette.size());

// A 5x7 bitmap font, one byte per column, bit 0 = top row. Only the glyphs
// the labels can contain are stored: A-Z, 0-9, space, percent and hyphen.
// Class names are upper-cased before drawing so no lower-case set is needed.
//
// Embedded glyphs avoid a font file or rendering-library dependency.
const int kGlyphWidth = 5;
const int kGlyphHeight = 7;

const std::array<Glyph, 10> kFontDigits = {{
    {{0x3E, 0x51, 0x49, 0x45, 0x3E}},
    {{0x00, 0x42, 0x7F, 0x40, 0x00}},
    {{0x42, 0x61, 0x51, 0x49, 0x46}},
    {{0x21, 0x41, 0x45, 0x4B, 0x31}},
    {{0x18, 0x14, 0x12, 0x7F, 0x10}},
    {{0x27, 0x45, 0x45, 0x45, 0x39}},
    {{0x3C, 0x4A, 0x49, 0x49, 0x30}},
    {{0x01, 0x71, 0x09, 0x05, 0x03}},
    {{0x36, 0x49, 0x49, 0x49, 0x36}},
    {{0x06, 0x49, 0x49, 0x29, 0x1E}},
}};

const std::array<Glyph, 26> kFontUpper = {{
    {{0x7E, 0x11, 0x11, 0x11, 0x7E}}, {{0x7F, 0x49, 0x49, 0x49, 0x36}},
    {{0x3E, 0x41, 0x41, 0x41, 0x22}}, {{0x7F, 0x41, 0x41, 0x22, 0x1C}},
    {{0x7F, 0x49, 0x49, 0x49, 0x41}}, {{0x7F, 0x09, 0x09, 0x09, 0x01}},
    {{0x3E, 0x41, 0x49, 0x49, 0x7A}}, {{0x7F, 0x08, 0x08, 0x08, 0x7F}},
    {{0x00, 0x41, 0x7F, 0x41, 0x00}}, {{0x20, 0x40, 0x41, 0x3F, 0x01}},
    {{0x7F, 0x08, 0x14, 0x22, 0x41}}, {{0x7F, 0x40, 0x40, 0x40, 0x40}},
    {{0x7F, 0x02, 0x0C, 0x02, 0x7F}}, {{0x7F, 0x04, 0x08, 0x10, 0x7F}},
    {{0x3E, 0x41, 0x41, 0x41, 0x3E}}, {{0x7F, 0x09, 0x09, 0x09, 0x06}},
    {{0x3E, 0x41, 0x51, 0x21, 0x5E}}, {{0x7F, 0x09, 0x19, 0x29, 0x46}},
    {{0x46, 0x49, 0x49, 0x49, 0x31}}, {{0x01, 0x01, 0x7F, 0x01, 0x01}},
    {{0x3F, 0x40, 0x40, 0x40, 0x3F}}, {{0x1F, 0x20, 0x40, 0x20, 0x1F}},
    {{0x3F, 0x40, 0x38, 0x40, 0x3F}}, {{0x63, 0x14, 0x08, 0x14, 0x63}},
    {{0x07, 0x08, 0x70, 0x08, 0x07}}, {{0x61, 0x51, 0x49, 0x45, 0x43}},
}};

const Glyph kGlyphSpace = {{0x00, 0x00, 0x00, 0x00, 0x00}};
const Glyph kGlyphPercent = {{0x23, 0x13, 0x08, 0x64, 0x62}};
const Glyph kGlyphHyphen = {{0x08, 0x08, 0x08, 0x08, 0x08}};

/** @brief Column bitmap for one character, or spaces when unrepresentable. */
const Glyph& GlyphFor(char c) {
  if (c >= '0' && c <= '9')
    return kFontDigits[static_cast<std::size_t>(c - '0')];

  if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  if (c >= 'A' && c <= 'Z')
    return kFontUpper[static_cast<std::size_t>(c - 'A')];

  if (c == '%') return kGlyphPercent;

  if (c == '-') return kGlyphHyphen;

  return kGlyphSpace;
}

/** @brief "PERSON 92%" for a detection. */
std::string LabelText(const cctv_box& box) {
  auto percent = static_cast<int>(box.score * 100.0f + 0.5f);
  percent = std::max(0, std::min(100, percent));
  std::array<char, 64> buf = {};
  std::snprintf(buf.data(), buf.size(), "%s %d%%", CocoLabel(box.class_id),
                percent);
  return std::string(buf.data());
}

/**
 * @brief Text scale for a frame width, so labels stay readable at any size.
 *
 * At 640x480 this gives 2, i.e. 10x14 pixel characters.
 */
int TextScale(int frame_width) { return std::max(1, frame_width / 320); }

/** @brief Black or white, whichever contrasts with the band colour. */
const Rgb& TextColorFor(const Rgb& rgb) {
  static const Rgb kBlack = {{0u, 0u, 0u}};
  static const Rgb kWhite = {{255u, 255u, 255u}};
  // Rec. 601 luma; the palette is bright, so most bands take black text.
  const int luma = (299 * rgb[0] + 587 * rgb[1] + 114 * rgb[2]) / 1000;
  return luma > 140 ? kBlack : kWhite;
}

}  // namespace

const int Annotator::kPaletteSize = kPaletteCount;

int Annotator::Draw(Frame* frame, const cctv_box* boxes, int count) const {
  if (frame == nullptr || !frame->IsValid() || boxes == nullptr) return 0;

  int drawn = 0;
  for (int i = 0; i < count; ++i) drawn += DrawBox(frame, boxes[i], i);
  return drawn;
}

int Annotator::DrawBox(Frame* frame, const cctv_box& box, int index) const {
  // Cycle colors for detection counts larger than the palette.
  const Rgb& color = kPalette[static_cast<std::size_t>(
      (index % kPaletteCount + kPaletteCount) % kPaletteCount)];
  // Round outward so a thin detection still yields a visible rectangle.
  const auto x0 = static_cast<int>(std::floor(box.x));
  const auto y0 = static_cast<int>(std::floor(box.y));
  const auto x1 = static_cast<int>(std::ceil(box.x + box.w));
  const auto y1 = static_cast<int>(std::ceil(box.y + box.h));

  // Reject boxes that fall entirely outside the frame.
  if (x1 < 0 || y1 < 0 || x0 >= frame->width || y0 >= frame->height) return 0;

  // Clip edges before drawing so oversized boxes still have visible borders.
  const auto cx0 = std::max(0, std::min(x0, frame->width - 1));
  const auto cy0 = std::max(0, std::min(y0, frame->height - 1));
  const auto cx1 = std::max(0, std::min(x1, frame->width - 1));
  const auto cy1 = std::max(0, std::min(y1, frame->height - 1));

  int written = 0;
  for (int t = 0; t < kBorderThickness; ++t) {
    for (int x = cx0; x <= cx1; ++x) {
      written += SetPixel(frame, x, cy0 + t, color);
      written += SetPixel(frame, x, cy1 - t, color);
    }
    for (int y = cy0; y <= cy1; ++y) {
      written += SetPixel(frame, cx0 + t, y, color);
      written += SetPixel(frame, cx1 - t, y, color);
    }
  }

  written += DrawLabel(frame, box, cx0, cy0, color);
  return written > 0 ? 1 : 0;
}

int Annotator::FillRect(Frame* frame, int x0, int y0, int x1, int y1,
                        const std::array<uint8_t, 3>& rgb) const {
  int written = 0;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) written += SetPixel(frame, x, y, rgb);

  return written;
}

int Annotator::DrawLabel(Frame* frame, const cctv_box& box, int x0, int y0,
                         const std::array<uint8_t, 3>& rgb) const {
  const std::string text = LabelText(box);
  const int scale = TextScale(frame->width);
  const int pad = scale;
  const int char_w = (kGlyphWidth + 1) * scale;  // +1 column of tracking
  const int band_h = kGlyphHeight * scale + 2 * pad;
  const int band_w = static_cast<int>(text.size()) * char_w + 2 * pad;

  // Prefer sitting just above the rectangle. When the detection touches the
  // top of the frame there is no room, so the band drops inside the box --
  // otherwise the label for anything at the top edge would be invisible.
  int band_y = y0 - band_h;
  if (band_y < 0) band_y = y0;

  // Keep the band from running off the right edge; a label clipped mid-word
  // is worse than one shifted left to fit.
  int band_x = x0;
  if (band_x + band_w > frame->width)
    band_x = std::max(0, frame->width - band_w);

  int written = FillRect(frame, band_x, band_y, band_x + band_w - 1,
                         band_y + band_h - 1, rgb);

  const Rgb& ink = TextColorFor(rgb);
  int pen_x = band_x + pad;
  for (const char character : text) {
    written +=
        DrawGlyph(frame, GlyphFor(character), pen_x, band_y, pad, scale, ink);
    pen_x += char_w;
  }
  return written;
}

int Annotator::DrawGlyph(Frame* frame, const std::array<uint8_t, 5>& glyph,
                         int pen_x, int band_y, int pad, int scale,
                         const std::array<uint8_t, 3>& ink) const {
  int written = 0;
  for (int col = 0; col < kGlyphWidth; ++col) {
    for (int row = 0; row < kGlyphHeight; ++row) {
      if ((glyph[col] & (1u << row)) == 0u) continue;

      const int glyph_x = pen_x + col * scale;
      const int glyph_y = band_y + pad + row * scale;
      written += FillRect(frame, glyph_x, glyph_y, glyph_x + scale - 1,
                          glyph_y + scale - 1, ink);
    }
  }
  return written;
}

int Annotator::SetPixel(Frame* frame, int x, int y,
                        const std::array<uint8_t, 3>& rgb) const {
  if (x < 0 || y < 0 || x >= frame->width || y >= frame->height) return 0;

  const std::size_t i =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(frame->width) +
       static_cast<std::size_t>(x)) *
      3u;
  frame->rgb[i] = rgb[0];
  frame->rgb[i + 1] = rgb[1];
  frame->rgb[i + 2] = rgb[2];
  return 3;
}

}  // namespace app
}  // namespace mep_cctv
