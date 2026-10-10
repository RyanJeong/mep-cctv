// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file bundled_paths_test.cc
 * @brief Tests for app/bundled_paths.cc.
 */

#include "app/bundled_paths.hpp"

#include <sys/stat.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "gtest/gtest.h"

namespace {

// The deployed artifact is one self-contained folder that gets copied to a Pi
// and run from anywhere, so every asset has to be located relative to the
// executable rather than to the working directory.

/** @brief Create a temporary directory tree for one test, and remove it. */
class BundledPathsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char tmpl[] = "/tmp/cctv_paths_XXXXXX";
    const char* const made = mkdtemp(tmpl);
    ASSERT_NE(nullptr, made);
    root_ = made;
    fake_exe_ = root_ + "/cctv_app";
  }

  void TearDown() override {
    // Depth-first removal; the tree is at most two levels deep.
    std::remove((root_ + "/static/index.html").c_str());
    std::remove((root_ + "/static").c_str());
    std::remove((root_ + "/yolov8n.onnx").c_str());
    std::remove(root_.c_str());
  }

  /** @brief Write an empty file, creating it if absent. */
  void Touch(const std::string& path) const {
    const std::ofstream out(path.c_str());
    ASSERT_TRUE(out.good());
  }

  std::string root_;
  std::string fake_exe_;
};

TEST_F(BundledPathsTest, ExecutableDirectoryStripsTheFileName) {
  EXPECT_EQ("/opt/cctv", mep_cctv::app::ExecutableDirectory("/opt/cctv/app"));
}

TEST_F(BundledPathsTest, ExecutableDirectoryOfBareNameIsCurrentDirectory) {
  EXPECT_EQ(".", mep_cctv::app::ExecutableDirectory("cctv_app"));
}

TEST_F(BundledPathsTest, ExecutableDirectoryHandlesNull) {
  EXPECT_EQ(".", mep_cctv::app::ExecutableDirectory(nullptr));
}

TEST_F(BundledPathsTest, FindsModelBesideTheExecutable) {
  Touch(root_ + "/yolov8n.onnx");
  EXPECT_EQ(root_ + "/yolov8n.onnx",
            mep_cctv::app::FindBundledModel(fake_exe_.c_str()));
}

TEST_F(BundledPathsTest, ReportsNoModelWhenAbsent) {
  EXPECT_EQ("", mep_cctv::app::FindBundledModel(fake_exe_.c_str()));
}

TEST_F(BundledPathsTest, FindsStaticDirectoryBesideTheExecutable) {
  ASSERT_EQ(0, mkdir((root_ + "/static").c_str(), 0755));
  Touch(root_ + "/static/index.html");
  EXPECT_EQ(root_ + "/static",
            mep_cctv::app::FindBundledStaticDir(fake_exe_.c_str()));
}

// A bare directory is not enough: an empty static/ would mount cleanly and
// then serve 404 for every page, which is what a missing index.html looks
// like from the browser.
TEST_F(BundledPathsTest, RejectsStaticDirectoryWithoutIndex) {
  ASSERT_EQ(0, mkdir((root_ + "/static").c_str(), 0755));
  EXPECT_EQ("", mep_cctv::app::FindBundledStaticDir(fake_exe_.c_str()));
}

TEST_F(BundledPathsTest, ReportsNoStaticDirectoryWhenAbsent) {
  EXPECT_EQ("", mep_cctv::app::FindBundledStaticDir(fake_exe_.c_str()));
}

}  // namespace
