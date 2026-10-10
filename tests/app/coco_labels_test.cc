// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file coco_labels_test.cc
 * @brief Tests for app/coco_labels.cc.
 */

#include "app/coco_labels.hpp"

#include "gtest/gtest.h"

namespace {

TEST(CocoLabelTest, KnownClassHasItsName) {
  EXPECT_STREQ("person", mep_cctv::app::CocoLabel(0));
  EXPECT_STREQ("car", mep_cctv::app::CocoLabel(2));
}

TEST(CocoLabelTest, LastClassIsInRange) {
  EXPECT_STREQ("toothbrush",
               mep_cctv::app::CocoLabel(mep_cctv::app::kCocoClassCount - 1));
}

TEST(CocoLabelTest, TableHasEightyClasses) {
  EXPECT_EQ(80, mep_cctv::app::kCocoClassCount);
}

// A corrupt or unexpected model must not index out of bounds.
TEST(CocoLabelTest, OutOfRangeIsSafe) {
  EXPECT_STREQ("unknown", mep_cctv::app::CocoLabel(-1));
  EXPECT_STREQ("unknown",
               mep_cctv::app::CocoLabel(mep_cctv::app::kCocoClassCount));
  EXPECT_STREQ("unknown", mep_cctv::app::CocoLabel(999999));
}

}  // namespace
