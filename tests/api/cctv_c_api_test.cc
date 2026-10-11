// Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

/**
 * @file c_api_test.cc
 * @brief Unit tests for the public C ABI (cctv.h).
 *
 * Tests cover:
 *   - cctv_engine_create: valid config, NULL config
 *   - cctv_engine_destroy: valid engine, NULL engine
 *   - cctv_detect_objects: null args -> CCTV_ERR_INVALID_ARG
 *   - cctv_detect_objects: zero dims -> CCTV_ERR_INVALID_ARG
 *   - cctv_detect_objects: synthetic image -> boxes returned + freed
 *   - cctv_free_detections: NULL -> CCTV_OK
 *   - cctv_detect_audio: null args -> CCTV_ERR_INVALID_ARG
 *   - cctv_detect_audio: valid samples -> result
 *   - cctv_version: non-empty string
 *   - cctv_backend: known string
 *   - cctv_simd_enabled: 0 or 1
 */

#include <cstdint>
#include <cstring>
#include <vector>

#include "cctv/cctv.h"
#include "gtest/gtest.h"

#if defined(CCTV_USE_GENERATED_MODEL) && CCTV_USE_GENERATED_MODEL
#include "src/backend_native/model_tvm.hpp"
#include "src/core/image_ops.hpp"
#include "src/core/yolo_postprocess.hpp"
#endif

namespace {  // NOSONAR - anonymous ns is the C++ idiom for file-local linkage

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class CApiTest : public ::testing::Test {  // NOSONAR - test fixture
 protected:
  void SetUp() override {
    cfg_.yolo_model_path = nullptr;
    cfg_.conf_threshold = 0.25f;
    cfg_.iou_threshold = 0.45f;
    engine_ = cctv_engine_create(&cfg_);
    // ONNX engines require a model path; native engines accept null paths.
    if (engine_ == nullptr) {
      GTEST_SKIP() << "engine create returned null (dl backend, needs models)";
    }
  }

  void TearDown() override {
    if (engine_ != nullptr) {
      cctv_engine_destroy(engine_);
      engine_ = nullptr;
    }
  }

  cctv_config cfg_{};
  cctv_engine* engine_ = nullptr;
};

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

TEST(CApiLifecycleTest, CreateWithNullConfigReturnsNull) {
  EXPECT_EQ(nullptr, cctv_engine_create(nullptr));
}

TEST(CApiLifecycleTest, CreateAndDestroyClean) {
  cctv_config cfg{};
  cfg.conf_threshold = 0.3f;
  cfg.iou_threshold = 0.5f;
  cctv_engine* engine = cctv_engine_create(&cfg);
  if (engine == nullptr) {
    // ONNX engines require a configured model.
    GTEST_SKIP() << "engine create returned null (dl backend, needs models)";
  }
  EXPECT_EQ(CCTV_OK, cctv_engine_destroy(engine));
}

TEST(CApiLifecycleTest, DestroyNullReturnsOk) {
  EXPECT_EQ(CCTV_OK, cctv_engine_destroy(nullptr));
}

// ---------------------------------------------------------------------------
// cctv_detect_objects -- argument validation
// ---------------------------------------------------------------------------

TEST_F(CApiTest, DetectObjectsNullEngineReturnsInvalidArg) {
  uint8_t rgb[3] = {0, 0, 0};
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(nullptr, rgb, 1, 1, &out));
}

TEST_F(CApiTest, DetectObjectsNullRgbReturnsInvalidArg) {
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, nullptr, 1, 1, &out));
}

TEST_F(CApiTest, DetectObjectsNullOutReturnsInvalidArg) {
  uint8_t rgb[3] = {0, 0, 0};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, rgb, 1, 1, nullptr));
}

TEST_F(CApiTest, DetectObjectsZeroWidthReturnsInvalidArg) {
  uint8_t rgb[3] = {0, 0, 0};
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, rgb, 0, 1, &out));
}

TEST_F(CApiTest, DetectObjectsZeroHeightReturnsInvalidArg) {
  uint8_t rgb[3] = {0, 0, 0};
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, rgb, 1, 0, &out));
}

TEST_F(CApiTest, DetectObjectsNegativeWidthReturnsInvalidArg) {
  uint8_t rgb[3] = {0, 0, 0};
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, rgb, -1, 1, &out));
}

// ---------------------------------------------------------------------------
// Security: oversized dimensions must be rejected (integer-overflow guard).
// ---------------------------------------------------------------------------

TEST_F(CApiTest, DetectObjectsOversizedWidthReturnsInvalidArg) {
  // 70000 > kMaxImageDim=65535; must not attempt allocation or index math.
  uint8_t rgb[3] = {0, 0, 0};
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, rgb, 70000, 1, &out));
}

TEST_F(CApiTest, DetectObjectsOversizedHeightReturnsInvalidArg) {
  uint8_t rgb[3] = {0, 0, 0};
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, rgb, 1, 70000, &out));
}

TEST_F(CApiTest, DetectObjectsBothDimsOversizedReturnsInvalidArg) {
  uint8_t rgb[3] = {0, 0, 0};
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, rgb, 70000, 70000, &out));
}

TEST_F(CApiTest, DetectObjectsAtMaxDimIsAccepted) {
  // kMaxImageDim itself must be accepted without crashing.
  // Allocating a 65535x65535 RGB buffer (~12 GB) is impractical in a test;
  // instead test kMaxImageDim x 1 (a single-row extreme-width image).
  const int W = 65535;
  const int H = 1;
  std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3, 128u);
  cctv_detections out{};
  const cctv_status status =
      cctv_detect_objects(engine_, img.data(), W, H, &out);
  // Must return CCTV_OK (not CCTV_ERR_INVALID_ARG).
  EXPECT_EQ(CCTV_OK, status);
  EXPECT_GE(out.count, 0);
  EXPECT_EQ(CCTV_OK, cctv_free_detections(&out));
}

TEST_F(CApiTest, DetectObjectsJustAboveMaxDimIsRejected) {
  uint8_t rgb[3] = {0, 0, 0};
  cctv_detections out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_objects(engine_, rgb, 65536, 1, &out));
}

// ---------------------------------------------------------------------------
// cctv_detect_objects -- functional (native backend placeholder)
// ---------------------------------------------------------------------------

TEST_F(CApiTest, DetectObjectsSyntheticImageIsWellFormed) {
  // Create a minimal synthetic image.
  const int W = 320;
  const int H = 240;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 128);

  cctv_detections out{};
  out.boxes = nullptr;
  out.count = 0;

  const cctv_status status =
      cctv_detect_objects(engine_, img.data(), W, H, &out);
  EXPECT_EQ(CCTV_OK, status);

  // Only the native placeholder guarantees zero detections for every input.
#if defined(CCTV_USE_NATIVE_BACKEND) && CCTV_USE_NATIVE_BACKEND && \
    !(defined(CCTV_USE_GENERATED_MODEL) && CCTV_USE_GENERATED_MODEL)
  EXPECT_EQ(0, out.count);
#else
  // Real-model and fixed-fixture outputs have different count expectations.
  EXPECT_GE(out.count, 0);
#endif
  if (out.count > 0) {
    ASSERT_NE(nullptr, out.boxes);
    for (int i = 0; i < out.count; ++i) {
      EXPECT_GE(out.boxes[i].score, 0.0f);
      EXPECT_LE(out.boxes[i].score, 1.0f);
      EXPECT_GE(out.boxes[i].w, 0.0f);
      EXPECT_GE(out.boxes[i].h, 0.0f);
    }
  }

  EXPECT_EQ(CCTV_OK, cctv_free_detections(&out));
  EXPECT_EQ(nullptr, out.boxes);
  EXPECT_EQ(0, out.count);
}

TEST_F(CApiTest, DetectObjectsAllBlackImageOk) {
  // All-black image: conf threshold may suppress all -> count can be 0.
  const int W = 64;
  const int H = 64;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 0);

  cctv_detections out{};
  const cctv_status status =
      cctv_detect_objects(engine_, img.data(), W, H, &out);
  EXPECT_EQ(CCTV_OK, status);
  // Zero detections are valid for a black frame.
  EXPECT_GE(out.count, 0);
  EXPECT_EQ(CCTV_OK, cctv_free_detections(&out));
}

// ---------------------------------------------------------------------------
// Source frames larger than the model input must remain within buffer bounds.
// ---------------------------------------------------------------------------

TEST_F(CApiTest, DetectObjects720pLandscapeNoCrash) {
  const int W = 1280;
  const int H = 720;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 128);

  cctv_detections out{};
  const cctv_status status =
      cctv_detect_objects(engine_, img.data(), W, H, &out);
  EXPECT_EQ(CCTV_OK, status);
  EXPECT_GE(out.count, 0);

  // Boxes must be mapped to original image space: within [0, W] x [0, H].
  for (int i = 0; i < out.count; ++i) {
    EXPECT_GE(out.boxes[i].x, 0.0f);
    EXPECT_GE(out.boxes[i].y, 0.0f);
    EXPECT_LE(out.boxes[i].x + out.boxes[i].w,
              static_cast<float>(W) + 1.0f);  // +1 for float rounding
    EXPECT_LE(out.boxes[i].y + out.boxes[i].h, static_cast<float>(H) + 1.0f);
  }
  EXPECT_EQ(CCTV_OK, cctv_free_detections(&out));
}

TEST_F(CApiTest, DetectObjectsTallFrameNoCrash) {
  // Regression: tall frame 480x1280 -- height > 640.
  const int W = 480;
  const int H = 1280;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 64);

  cctv_detections out{};
  const cctv_status status =
      cctv_detect_objects(engine_, img.data(), W, H, &out);
  EXPECT_EQ(CCTV_OK, status);
  EXPECT_GE(out.count, 0);

  for (int i = 0; i < out.count; ++i) {
    EXPECT_GE(out.boxes[i].x, 0.0f);
    EXPECT_GE(out.boxes[i].y, 0.0f);
    EXPECT_LE(out.boxes[i].x + out.boxes[i].w, static_cast<float>(W) + 1.0f);
    EXPECT_LE(out.boxes[i].y + out.boxes[i].h, static_cast<float>(H) + 1.0f);
  }
  EXPECT_EQ(CCTV_OK, cctv_free_detections(&out));
}

TEST_F(CApiTest, DetectObjectsFullHdNoCrash) {
  // 1920x1080: another common camera resolution.
  const int W = 1920;
  const int H = 1080;
  std::vector<uint8_t> img(static_cast<unsigned>(W * H * 3), 200);

  cctv_detections out{};
  const cctv_status status =
      cctv_detect_objects(engine_, img.data(), W, H, &out);
  EXPECT_EQ(CCTV_OK, status);
  EXPECT_GE(out.count, 0);
  EXPECT_EQ(CCTV_OK, cctv_free_detections(&out));
}

// ---------------------------------------------------------------------------
// cctv_free_detections
// ---------------------------------------------------------------------------

TEST(CApiFreeTest, FreeNullReturnsOk) {
  EXPECT_EQ(CCTV_OK, cctv_free_detections(nullptr));
}

TEST(CApiFreeTest, FreeAlreadyFreedStructOk) {
  cctv_detections dets{};
  dets.boxes = nullptr;
  dets.count = 0;
  EXPECT_EQ(CCTV_OK, cctv_free_detections(&dets));
}

// ---------------------------------------------------------------------------
// cctv_detect_audio -- argument validation
// ---------------------------------------------------------------------------

TEST_F(CApiTest, DetectAudioNullEngineReturnsInvalidArg) {
  float samples[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  cctv_audio_result out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_audio(nullptr, samples, 4, 16000, &out));
}

TEST_F(CApiTest, DetectAudioNullSamplesReturnsInvalidArg) {
  cctv_audio_result out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_audio(engine_, nullptr, 4, 16000, &out));
}

TEST_F(CApiTest, DetectAudioNullOutReturnsInvalidArg) {
  float samples[4] = {0.0f};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_audio(engine_, samples, 4, 16000, nullptr));
}

TEST_F(CApiTest, DetectAudioZeroSamplesReturnsInvalidArg) {
  float samples[1] = {0.0f};
  cctv_audio_result out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_audio(engine_, samples, 0, 16000, &out));
}

TEST_F(CApiTest, DetectAudioZeroSampleRateReturnsInvalidArg) {
  float samples[4] = {0.0f};
  cctv_audio_result out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_audio(engine_, samples, 4, 0, &out));
}

// Security: oversized num_samples must be rejected (integer-overflow guard).
TEST_F(CApiTest, DetectAudioOversizedSamplesReturnsInvalidArg) {
  // kMaxAudioSamples=120000000; pass kMaxAudioSamples+1.
  float samples[1] = {0.0f};
  cctv_audio_result out{};
  EXPECT_EQ(CCTV_ERR_INVALID_ARG,
            cctv_detect_audio(engine_, samples, 120000001, 16000, &out));
}

// ---------------------------------------------------------------------------
// cctv_detect_audio -- functional
// ---------------------------------------------------------------------------

TEST_F(CApiTest, DetectAudioSilencePersonNotPresent) {
  // All-zero audio -> RMS = 0 -> person_present = 0.
  const int N = 1600;
  std::vector<float> samples(static_cast<unsigned>(N), 0.0f);
  cctv_audio_result out{};
  EXPECT_EQ(CCTV_OK,
            cctv_detect_audio(engine_, samples.data(), N, 16000, &out));
  EXPECT_EQ(0, out.person_present);
  EXPECT_NEAR(0.0f, out.score, 1e-5f);
}

TEST_F(CApiTest, DetectAudioLoudSignalPersonPresent) {
  // An ALTERNATING signal, not a constant one: a constant 0.5 is pure DC
  // offset, which the detector deliberately ignores (a biased microphone is
  // not a noisy room). Real sound varies, so the test signal must too.
  const int N = 1600;
  std::vector<float> samples(static_cast<unsigned>(N), 0.0f);
  for (int i = 0; i < N; ++i) {
    samples[static_cast<unsigned>(i)] = (i % 2 == 0) ? 0.5f : -0.5f;
  }
  cctv_audio_result out{};
  EXPECT_EQ(CCTV_OK,
            cctv_detect_audio(engine_, samples.data(), N, 16000, &out));
  EXPECT_EQ(1, out.person_present);
  EXPECT_GT(out.score, 0.0f);
}

TEST_F(CApiTest, DetectAudioScoreInValidRange) {
  const int N = 800;
  std::vector<float> samples(static_cast<unsigned>(N), 0.1f);
  cctv_audio_result out{};
  EXPECT_EQ(CCTV_OK, cctv_detect_audio(engine_, samples.data(), N, 8000, &out));
  EXPECT_GE(out.score, 0.0f);
}

// ---------------------------------------------------------------------------
// Metadata getters
// ---------------------------------------------------------------------------

TEST(CApiMetaTest, VersionNonEmpty) {
  const char* v = cctv_version();
  ASSERT_NE(nullptr, v);
  EXPECT_GT(std::strlen(v), 0u);
}

TEST(CApiMetaTest, VersionMatchesProjectVersion) {
  const char* v = cctv_version();
  ASSERT_NE(nullptr, v);
  // Version should be a dotted string like "0.1.0".
  EXPECT_NE(nullptr, std::strchr(v, '.'));
}

TEST(CApiMetaTest, BackendNonEmpty) {
  const char* b = cctv_backend();
  ASSERT_NE(nullptr, b);
  EXPECT_GT(std::strlen(b), 0u);
}

TEST(CApiMetaTest, BackendIsKnownValue) {
  const char* b = cctv_backend();
  ASSERT_NE(nullptr, b);
  // The reported name is the mode name, spelled exactly as the docs, the
  // artifact folders and CCTV_MODE spell it.
  const bool is_native = std::strcmp(b, "native") == 0;
  const bool is_dl = std::strcmp(b, "onnx") == 0;
  EXPECT_TRUE(is_native || is_dl);
}

TEST(CApiMetaTest, SimdEnabledIsBinaryFlag) {
  const int s = cctv_simd_enabled();
  EXPECT_TRUE(s == 0 || s == 1);
}

TEST(CApiMetaTest, SimdEnabledMatchesCompileFlag) {
#if defined(CCTV_SIMD_ENABLED) && CCTV_SIMD_ENABLED
  EXPECT_EQ(1, cctv_simd_enabled());
#else
  EXPECT_EQ(0, cctv_simd_enabled());
#endif
}

// ---------------------------------------------------------------------------
// TVM-compiled model -- only built when the real model is linked in
// ---------------------------------------------------------------------------

#if defined(CCTV_USE_GENERATED_MODEL) && CCTV_USE_GENERATED_MODEL

// Derived from the model geometry so these hold at any kInputSize.
constexpr unsigned kExpectedOutputElements = static_cast<unsigned>(
    mep_cctv::core::kOutputStride * mep_cctv::core::kNumAnchors);

TEST(GeneratedModelTest, ForwardOutputHasYoloShape) {
  const std::vector<float> input(
      static_cast<unsigned>(mep_cctv::core::kInputElements), 0.5f);
  const std::vector<float> out = mep_cctv::backend_native::generated::Forward(
      input.data(), static_cast<int>(input.size()));
  EXPECT_EQ(kExpectedOutputElements, out.size());
}

TEST(GeneratedModelTest, ForwardRejectsWrongElementCount) {
  const std::vector<float> input(16, 0.5f);
  const std::vector<float> out =
      mep_cctv::backend_native::generated::Forward(input.data(), 16);
  // Contract: a well-formed all-zero tensor rather than a crash.
  ASSERT_EQ(kExpectedOutputElements, out.size());
  EXPECT_FLOAT_EQ(0.0f, out[0]);
}

TEST(GeneratedModelTest, ForwardRejectsNullInput) {
  const std::vector<float> out = mep_cctv::backend_native::generated::Forward(
      nullptr, mep_cctv::core::kInputElements);
  EXPECT_EQ(kExpectedOutputElements, out.size());
}

// The whole point of replacing the placeholder: the placeholder did
// `(void)input_nchw;` and returned a fixed tensor, so it would FAIL this test.
// Passing it proves the compiled network actually reads the pixels.
TEST(GeneratedModelTest, OutputDependsOnInput) {
  const int n = mep_cctv::core::kInputElements;
  std::vector<float> dark(static_cast<unsigned>(n), 0.0f);
  std::vector<float> bright(static_cast<unsigned>(n), 1.0f);

  const std::vector<float> a =
      mep_cctv::backend_native::generated::Forward(dark.data(), n);
  const std::vector<float> b =
      mep_cctv::backend_native::generated::Forward(bright.data(), n);

  ASSERT_EQ(a.size(), b.size());
  bool differs = false;
  for (size_t i = 0; i < a.size() && !differs; ++i) {
    differs = (a[i] != b[i]);
  }
  EXPECT_TRUE(differs) << "output identical for different inputs -- the model "
                          "is ignoring its input";
}

TEST(GeneratedModelTest, RepeatedCallsAreDeterministic) {
  const int n = mep_cctv::core::kInputElements;
  const std::vector<float> input(static_cast<unsigned>(n), 0.25f);
  const std::vector<float> first =
      mep_cctv::backend_native::generated::Forward(input.data(), n);
  const std::vector<float> second =
      mep_cctv::backend_native::generated::Forward(input.data(), n);
  EXPECT_EQ(first, second);
}

#endif  // CCTV_USE_GENERATED_MODEL

}  // namespace
