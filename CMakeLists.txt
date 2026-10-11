# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

cmake_minimum_required(VERSION 4.0)
project(mep_cctv VERSION 0.1.0 LANGUAGES CXX C)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Export the compilation database for editor and analysis tools.
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# User-facing options
# ---------------------------------------------------------------------------
set(CCTV_MODE "native" CACHE STRING "Build mode: native or onnx")
set_property(CACHE CCTV_MODE PROPERTY STRINGS native onnx)

# Where the native mode gets its model:
#   auto        use the TVM-generated sources when present, else the stand-in
#   real        require them; fail the configure if they are absent
#   placeholder ignore them and link the stand-in
#
set(CCTV_MODEL "auto" CACHE STRING "native model source: auto, real or placeholder")
set_property(CACHE CCTV_MODEL PROPERTY STRINGS auto real placeholder)

# Scalar kernels provide the portable reference configuration.
option(CCTV_SCALAR_KERNELS
  "Advanced: build the scalar reference kernels instead of SIMD" OFF)
mark_as_advanced(CCTV_SCALAR_KERNELS)
option(BUILD_TESTS "Build unit tests (requires network for FetchContent)" ON)
option(BUILD_APP "Build the cctv_app CCTV server (requires network for FetchContent)" ON)
option(ENABLE_COVERAGE "Instrument for lcov/llvm-cov coverage" OFF)
option(ENABLE_ASAN "AddressSanitizer + UBSan + LeakSan" OFF)
option(ENABLE_TSAN "ThreadSanitizer" OFF)
option(CCTV_PORTABLE_RUNTIME
  "Link the C++ runtime statically so the artifact runs on any Pi OS release"
  OFF)
# ---------------------------------------------------------------------------
# Compatibility with the previous option names
# ---------------------------------------------------------------------------
# Translate legacy cache entries once; native-only options apply only to native.
if(DEFINED CCTV_BACKEND)
  if(CCTV_BACKEND STREQUAL "dl")
    set(CCTV_MODE "onnx" CACHE STRING "Build mode" FORCE)
  else()
    set(CCTV_MODE "${CCTV_BACKEND}" CACHE STRING "Build mode" FORCE)
  endif()
  unset(CCTV_BACKEND CACHE)
  message(STATUS "legacy CCTV_BACKEND -> CCTV_MODE=${CCTV_MODE} (cache entry removed)")
endif()

if(DEFINED CCTV_USE_GENERATED_MODEL)
  if(CCTV_MODE STREQUAL "native")
    if(CCTV_USE_GENERATED_MODEL)
      set(CCTV_MODEL "real" CACHE STRING "native model source" FORCE)
    else()
      set(CCTV_MODEL "placeholder" CACHE STRING "native model source" FORCE)
    endif()
  endif()
  unset(CCTV_USE_GENERATED_MODEL CACHE)
  message(STATUS "legacy CCTV_USE_GENERATED_MODEL -> CCTV_MODEL (cache entry removed)")
endif()

if(DEFINED CCTV_SIMD)
  if(CCTV_MODE STREQUAL "native" AND NOT CCTV_SIMD)
    set(CCTV_SCALAR_KERNELS ON CACHE BOOL "scalar reference kernels" FORCE)
  endif()
  unset(CCTV_SIMD CACHE)
  message(STATUS "legacy CCTV_SIMD -> CCTV_SCALAR_KERNELS (cache entry removed)")
endif()

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if(NOT CCTV_MODE MATCHES "^(native|onnx)$")
  message(FATAL_ERROR "CCTV_MODE must be native or onnx (got '${CCTV_MODE}').")
endif()
if(NOT CCTV_MODEL MATCHES "^(auto|real|placeholder)$")
  message(FATAL_ERROR
    "CCTV_MODEL must be auto, real or placeholder (got '${CCTV_MODEL}').")
endif()

# Reset native-only options when switching an existing cache to onnx.
if(CCTV_MODE STREQUAL "onnx")
  set(CCTV_MODEL "auto" CACHE STRING "native model source" FORCE)
  set(CCTV_SCALAR_KERNELS OFF CACHE BOOL "scalar reference kernels" FORCE)
endif()

# ---------------------------------------------------------------------------
# Resolve CCTV_MODEL into a single boolean the rest of the file can use
# ---------------------------------------------------------------------------
# Resolve model selection from generated files and available runtime settings.
set(CCTV_GENERATED_DIR "${CMAKE_SOURCE_DIR}/src/backend_native/generated")
set(CCTV_MODEL_PREFIX "yolov8n_tuned")
set(CCTV_MODEL_USE_REAL OFF)

if(CCTV_MODE STREQUAL "native")
  set(_cctv_model_present TRUE)
  foreach(_gen "${CCTV_MODEL_PREFIX}_kernels.o" "${CCTV_MODEL_PREFIX}_blob.cc")
    if(NOT EXISTS "${CCTV_GENERATED_DIR}/${_gen}")
      set(_cctv_model_present FALSE)
    endif()
  endforeach()

  if(CCTV_MODEL STREQUAL "real")
    if(NOT _cctv_model_present)
      message(FATAL_ERROR
        "CCTV_MODEL=real but the TVM-generated sources are missing from "
        "${CCTV_GENERATED_DIR}. Apply the committed tuning database (about "
        "two minutes; see models/README.md):\n"
        "  .venv-convert/bin/python models/convert/tvm_compile.py \\\n"
        "    --onnx models/onnx/yolov8n.onnx \\\n"
        "    --input-name images --input-shape 1,3,320,320 \\\n"
        "    --out yolov8n_tuned --target arm64 \\\n"
        "    --emit-embed src/backend_native/generated \\\n"
        "    --tune-db models/tuning/rpi4\n"
        "Or configure with -DCCTV_MODEL=placeholder to use the stand-in.")
    endif()
    set(CCTV_MODEL_USE_REAL ON)
  elseif(CCTV_MODEL STREQUAL "auto")
    # Generated files alone are insufficient; auto also needs a runtime location.
    if(_cctv_model_present AND (CCTV_TVM_STATIC_ROOT OR CCTV_TVM_ROOT))
      set(CCTV_MODEL_USE_REAL ON)
    elseif(_cctv_model_present)
      message(STATUS
        "CCTV_MODEL=auto: generated sources found, but no TVM runtime was "
        "given (-DCCTV_TVM_STATIC_ROOT or -DCCTV_TVM_ROOT), so the stand-in "
        "model is used. Pass -DCCTV_MODEL=real to require the real one.")
    endif()
  elseif(NOT CCTV_MODEL STREQUAL "placeholder")
    message(FATAL_ERROR
      "CCTV_MODEL must be 'auto', 'real' or 'placeholder' (got "
      "'${CCTV_MODEL}').")
  endif()
endif()

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime root directory (optional override)")
set(ONNXRUNTIME_VERSION "1.18.1" CACHE STRING "ONNX Runtime release version")

if(ENABLE_ASAN AND ENABLE_TSAN)
  message(FATAL_ERROR "ENABLE_ASAN and ENABLE_TSAN are mutually exclusive")
endif()

# ---------------------------------------------------------------------------
# Portable runtime: one artifact for every 64-bit Raspberry Pi OS release
# ---------------------------------------------------------------------------
# Hide absorbed archive symbols to avoid runtime symbol interposition.
# glibc remains dynamic; see docker/README.md for compatibility limits.
if(CCTV_PORTABLE_RUNTIME)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR
      "CCTV_PORTABLE_RUNTIME is Linux-only (got ${CMAKE_SYSTEM_NAME}); "
      "macOS has no static libstdc++ to link.")
  endif()
  add_link_options(-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL)
  message(STATUS "Portable runtime: C++ runtime linked statically")
endif()

# ---------------------------------------------------------------------------
# Global warning flags
# ---------------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Werror -pedantic)

# Suppress a false pedantic warning on some GCC versions for C++ TUs
# that include C headers with flexible-array members.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wno-pedantic)
endif()

# ---------------------------------------------------------------------------
# Sanitizers
# ---------------------------------------------------------------------------
if(ENABLE_ASAN)
  if(APPLE)
    # Apple clang does not support -fsanitize=leak (only Linux libasan does).
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: ASan + UBSan (Apple: LeakSan unavailable)")
  else()
    add_compile_options(-fsanitize=address,undefined,leak -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined,leak)
    message(STATUS "Sanitizers: ASan + UBSan + LeakSan")
  endif()
endif()

if(ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
  message(STATUS "Sanitizers: TSan")
endif()

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate -fcoverage-mapping)
  else()
    add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
    add_link_options(--coverage)
  endif()
  message(STATUS "Coverage: enabled")
endif()

# ---------------------------------------------------------------------------
# SIMD detection and source selection
# ---------------------------------------------------------------------------
include(cmake/simd.cmake)

# ---------------------------------------------------------------------------
# Release optimizations
# ---------------------------------------------------------------------------
include(cmake/optimizations.cmake)

# ---------------------------------------------------------------------------
# Visibility: default hidden; API symbols exported explicitly
# ---------------------------------------------------------------------------
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ---------------------------------------------------------------------------
# Core sources (always compiled)
# ---------------------------------------------------------------------------
set(CCTV_CORE_SOURCES
  src/core/audio_detect.cc
  src/core/yolo_postprocess.cc
  src/api/cctv_c_api.cc
)

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
if(CCTV_MODE STREQUAL "native")
  if(CCTV_MODEL_USE_REAL)
    # Generated model files and runtime requirements are described in models/README.md.
    include(cmake/tvm.cmake)

    # The LLVM-generated object must match this build's target architecture.
    set(CCTV_KERNELS_OBJECT
      "${CCTV_GENERATED_DIR}/${CCTV_MODEL_PREFIX}_kernels.o")
    set_source_files_properties("${CCTV_KERNELS_OBJECT}"
      PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

    set(CCTV_BACKEND_SOURCES
      src/backend_native/native_backend.cc
      src/backend_native/model_tvm.cc
      "${CCTV_KERNELS_OBJECT}"
      "src/backend_native/generated/${CCTV_MODEL_PREFIX}_blob.cc"
    )
    set(CCTV_BACKEND_DEFINES
      CCTV_USE_NATIVE_BACKEND=1
      CCTV_USE_GENERATED_MODEL=1
      CCTV_BACKEND_NAME="native"
    )
    message(STATUS "CCTV backend: native (TVM-compiled YOLOv8n, autotuned)")
  else()
    set(CCTV_BACKEND_SOURCES
      src/backend_native/native_backend.cc
      src/backend_native/model_placeholder.cc
    )
    set(CCTV_BACKEND_DEFINES
      CCTV_USE_NATIVE_BACKEND=1
      CCTV_BACKEND_NAME="native"
    )
    message(STATUS "CCTV backend: native (placeholder -- reports no detections)")
  endif()

elseif(CCTV_MODE STREQUAL "onnx")
  # Fetch the pinned Linux runtime unless an installation is provided.
  if(NOT ONNXRUNTIME_ROOT)
    include(FetchContent)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
      set(ONNXRUNTIME_ARCH "aarch64")
      set(ONNXRUNTIME_SHA256
        "c1dcd8ab29e8d227d886b6ee415c08aea893956acf98f0758a42a84f27c02851")
    else()
      set(ONNXRUNTIME_ARCH "x64")
      set(ONNXRUNTIME_SHA256
        "a0994512ec1e1debc00c18bfc7a5f16249f6ebd6a6128ff2034464cc380ea211")
    endif()

    set(ONNXRUNTIME_URL
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz")
    FetchContent_Declare(onnxruntime
      URL "${ONNXRUNTIME_URL}"
      URL_HASH "SHA256=${ONNXRUNTIME_SHA256}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(onnxruntime)
    if(NOT onnxruntime_POPULATED)
      cmake_policy(PUSH)
      if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
      endif()
      FetchContent_Populate(onnxruntime)
      cmake_policy(POP)
    endif()
    set(ONNXRUNTIME_ROOT "${onnxruntime_SOURCE_DIR}")
    message(STATUS "CCTV dl backend: fetched ONNX Runtime from ${ONNXRUNTIME_URL}")
  endif()

  find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS
    "${ONNXRUNTIME_ROOT}/lib"
    "${ONNXRUNTIME_ROOT}/lib64")
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
    "${ONNXRUNTIME_ROOT}/include"
    "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES onnxruntime)
  if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR
      "CCTV_MODE=onnx requires ONNX Runtime, but it was not found. "
      "Install onnxruntime or use -DCCTV_MODE=native. "
      "You can also point CMake at an unpacked release with "
      "-DONNXRUNTIME_ROOT=/path/to/onnxruntime or "
      "-DCMAKE_PREFIX_PATH=/path/to/onnxruntime.")
  endif()
  set(CCTV_BACKEND_SOURCES
    src/backend_onnx/onnx_backend.cc
  )
  set(CCTV_BACKEND_DEFINES
    CCTV_USE_ONNX_BACKEND=1
    CCTV_BACKEND_NAME="onnx"
  )
  message(STATUS "CCTV mode: onnx (ONNX Runtime)")

else()
  message(FATAL_ERROR "CCTV_BACKEND must be 'native' or 'dl', got '${CCTV_BACKEND}'")
endif()

# ---------------------------------------------------------------------------
# Build-mode banner
# ---------------------------------------------------------------------------
if(CCTV_MODE STREQUAL "onnx")
  set(CCTV_MODE_NAME "onnx -- the original model, run by ONNX Runtime")
elseif(CCTV_MODEL_USE_REAL)
  set(CCTV_MODE_NAME "native -- TVM-compiled and fully optimized")
else()
  set(CCTV_MODE_NAME "native -- stand-in model, detects nothing (see models/README.md)")
endif()
message(STATUS "CCTV mode: ${CCTV_MODE_NAME}")

# ---------------------------------------------------------------------------
# Shared library target: cctv
# ---------------------------------------------------------------------------
add_library(cctv SHARED
  ${CCTV_CORE_SOURCES}
  ${CCTV_BACKEND_SOURCES}
  ${CCTV_SIMD_SOURCES}
)

target_include_directories(cctv
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

# Resolve bundled dependencies relative to the library.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(cctv PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
  set_target_properties(cctv PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path")
endif()

target_compile_definitions(cctv
  PRIVATE
  CCTV_BUILDING_SHARED=1
  CCTV_VERSION="${PROJECT_VERSION}"
  CCTV_SIMD_ENABLED=${CCTV_SIMD_ENABLED_VAL}
  ${CCTV_BACKEND_DEFINES}
)

# Apply SIMD compiler flags only to the SIMD source files.
if(CCTV_SIMD AND CCTV_SIMD_FLAGS)
  foreach(simd_src ${CCTV_SIMD_SOURCES})
    set_source_files_properties(${simd_src}
      PROPERTIES COMPILE_OPTIONS "${CCTV_SIMD_FLAGS}")
  endforeach()
endif()

# ---------------------------------------------------------------------------
# native backend with the real TVM model: headers, libraries, and the
# concessions that machine-generated code needs.
# ---------------------------------------------------------------------------
if(CCTV_MODE STREQUAL "native" AND CCTV_MODEL_USE_REAL)
  target_include_directories(cctv SYSTEM PRIVATE ${CCTV_TVM_INCLUDE_DIRS})
  # Static TVM parallel schedules require an explicit thread runtime.
  find_package(Threads REQUIRED)
  target_link_libraries(cctv PRIVATE ${CCTV_TVM_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})

  # Avoid optimizing the generated literal weight array.
  set_source_files_properties(
    "src/backend_native/generated/${CCTV_MODEL_PREFIX}_blob.cc"
    PROPERTIES COMPILE_OPTIONS "-w;-O0;-std=c++17")

  # tvm/ffi headers require C++17; the rest of the library stays C++14.
  set_source_files_properties(
    src/backend_native/model_tvm.cc
    PROPERTIES COMPILE_OPTIONS "-std=c++17")
endif()

# Link ONNX Runtime only for onnx mode.
if(CCTV_MODE STREQUAL "onnx")
  target_include_directories(cctv SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(cctv PRIVATE ${ONNXRUNTIME_LIB})
endif()

# ---------------------------------------------------------------------------
# CCTV application (cctv_app): capture -> inference -> annotate -> HTTP
# ---------------------------------------------------------------------------
if(BUILD_APP)
  include(FetchContent)

  # cpp-httplib: MIT, header-only HTTP/1.1 server. Chunked responses are what
  # make the multipart/x-mixed-replace MJPEG stream possible.
  FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.3
  )
  FetchContent_MakeAvailable(httplib)

  # stb_image_write: public domain, single header, in-memory JPEG encoding.
  # Pinned to a commit because the repository publishes no releases.
  FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
  )
  FetchContent_MakeAvailable(stb)

  # Select platform capture implementations without exposing platform headers.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CCTV_VIDEO_CAPTURE_SOURCE app/video_capture_v4l2.cc)
    find_library(ALSA_LIB asound)
    if(ALSA_LIB)
      set(CCTV_AUDIO_CAPTURE_SOURCE app/audio_capture_alsa.cc)
      message(STATUS "CCTV app: V4L2 camera + ALSA microphone")
    else()
      set(CCTV_AUDIO_CAPTURE_SOURCE app/audio_capture_null.cc)
      message(STATUS "CCTV app: V4L2 camera, no ALSA (install libasound2-dev)")
    endif()
  else()
    set(CCTV_VIDEO_CAPTURE_SOURCE app/video_capture_null.cc)
    set(CCTV_AUDIO_CAPTURE_SOURCE app/audio_capture_null.cc)
    message(STATUS "CCTV app: no capture backend for ${CMAKE_SYSTEM_NAME}; "
      "using synthetic video and silent audio")
  endif()

  add_library(cctv_app_core STATIC
    app/annotator.cc
    app/bundled_paths.cc
    app/coco_labels.cc
    app/detector.cc
    app/host_metrics.cc
    app/jpeg_encoder.cc
    app/pipeline.cc
    app/server.cc
    app/sources.cc
    ${CCTV_VIDEO_CAPTURE_SOURCE}
    ${CCTV_AUDIO_CAPTURE_SOURCE}
  )

  target_include_directories(cctv_app_core
    PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    PRIVATE
    ${stb_SOURCE_DIR}
  )

  target_link_libraries(cctv_app_core PUBLIC cctv httplib::httplib)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ALSA_LIB)
    target_link_libraries(cctv_app_core PUBLIC ${ALSA_LIB})
  endif()

  find_package(Threads REQUIRED)
  target_link_libraries(cctv_app_core PUBLIC Threads::Threads)

  # stb_image_write is third-party single-header code and does not survive
  # -Wall -Wextra -Werror -pedantic; scope the exemption to the one TU that
  # includes it rather than weakening the whole target.
  set_source_files_properties(app/jpeg_encoder.cc PROPERTIES COMPILE_OPTIONS "-w")

  add_executable(cctv_app app/main.cc)
  target_link_libraries(cctv_app PRIVATE cctv_app_core)

  # Build artifacts use a flat folder; installed binaries use ../lib.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(cctv_app PROPERTIES
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN/../lib"
      BUILD_WITH_INSTALL_RPATH FALSE)
  elseif(APPLE)
    set_target_properties(cctv_app PROPERTIES
      BUILD_RPATH "@loader_path"
      INSTALL_RPATH "@loader_path/../lib")
  endif()

  install(TARGETS cctv_app RUNTIME DESTINATION bin)
  install(DIRECTORY app/static DESTINATION share/mep_cctv)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  # Prevent GoogleTest from overriding our compiler/linker settings.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # GoogleTest v1.15 uses [[maybe_unused]] (C++17) and has character-conversion
  # warnings on some Clang versions.  Suppress these for gtest/gmock targets
  # only; our library and test sources stay fully C++14 pedantic.
  # Both flags are Clang-only: GCC does not know them, and applying a C++
  # warning flag to a C source is a hard error under -Werror.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    foreach(_gt_target gtest gtest_main gmock gmock_main)
      if(TARGET ${_gt_target})
        target_compile_options(${_gt_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-c++17-extensions>
        )
      endif()
    endforeach()
  endif()

  # Test executable.
  # tests/ mirrors the source tree: src/core/nms.cc -> tests/core/nms_test.cc.
  add_executable(cctv_tests
    tests/core/audio_detect_test.cc
    tests/core/image_ops_test.cc
    tests/core/nms_test.cc
    tests/core/yolo_postprocess_test.cc
    tests/api/cctv_c_api_test.cc
  )

  # onnx-specific tests compiled only for that mode.
  if(CCTV_MODE STREQUAL "onnx")
    target_sources(cctv_tests PRIVATE tests/backend_onnx/onnx_backend_test.cc)
  endif()

  # App tests need the app library, so they only exist when the app is built.
  if(BUILD_APP)
    target_sources(cctv_tests PRIVATE
      tests/app/annotator_test.cc
      tests/app/bundled_paths_test.cc
      tests/app/coco_labels_test.cc
      tests/app/detector_test.cc
      tests/app/frame_test.cc
      tests/app/host_metrics_test.cc
      tests/app/jpeg_encoder_test.cc
      tests/app/pipeline_test.cc
      tests/app/sources_test.cc
    )
    target_link_libraries(cctv_tests PRIVATE cctv_app_core)
  endif()

  target_include_directories(cctv_tests
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
  )

  if(CCTV_MODE STREQUAL "onnx")
    target_include_directories(cctv_tests SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  endif()

  target_compile_definitions(cctv_tests
    PRIVATE
    CCTV_VERSION="${PROJECT_VERSION}"
    CCTV_SIMD_ENABLED=${CCTV_SIMD_ENABLED_VAL}
    ${CCTV_BACKEND_DEFINES}
    FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/onnx"
  )

  # The test binary compiles core + backend sources directly so it does not
  # depend on exported C++ symbols from libcctv (hidden visibility).
  # Only one copy of each compilation unit is included (SIMD or scalar,
  # matching the build configuration) to avoid ODR violations.
  if(CCTV_MODE STREQUAL "native")
    target_sources(cctv_tests PRIVATE
      ${CCTV_SIMD_SOURCES} # image_ops_{scalar,simd}.cc + nms_{scalar,simd}.cc
      src/core/audio_detect.cc
      src/core/yolo_postprocess.cc
      src/backend_native/native_backend.cc
    )
    # Mirror the library's model choice so the tests exercise whichever
    # implementation was actually built.  The per-source compile options set
    # above are directory-scoped, so they apply to this target too.
    if(CCTV_MODEL_USE_REAL)
      target_sources(cctv_tests PRIVATE
        src/backend_native/model_tvm.cc
        "${CCTV_KERNELS_OBJECT}"
        "src/backend_native/generated/${CCTV_MODEL_PREFIX}_blob.cc"
        src/api/cctv_c_api.cc
      )
      target_include_directories(cctv_tests SYSTEM PRIVATE ${CCTV_TVM_INCLUDE_DIRS})
      find_package(Threads REQUIRED)
      target_link_libraries(cctv_tests PRIVATE ${CCTV_TVM_LIBRARIES}
        Threads::Threads ${CMAKE_DL_LIBS})
      # Keep one TVM registry/runtime copy per test process.
      set(CCTV_TESTS_SKIP_LIBCCTV TRUE)
    else()
      target_sources(cctv_tests PRIVATE
        src/backend_native/model_placeholder.cc
        tests/backend_native/model_placeholder_test.cc
      )
    endif()
  else()
    # Compile ONNX/core internals directly for hidden-symbol tests.
    target_sources(cctv_tests PRIVATE
      ${CCTV_SIMD_SOURCES}
      src/core/audio_detect.cc
      src/core/yolo_postprocess.cc
      src/backend_onnx/onnx_backend.cc
    )
  endif()

  # Apply SIMD flags to the simd sources in the test binary.
  if(CCTV_SIMD AND CCTV_SIMD_ENABLED_VAL)
    foreach(simd_src
      "${CMAKE_SOURCE_DIR}/src/core/image_ops_simd.cc"
      "${CMAKE_SOURCE_DIR}/src/core/nms_simd.cc")
      set_source_files_properties(${simd_src}
        PROPERTIES COMPILE_OPTIONS "${CCTV_SIMD_FLAGS}")
    endforeach()
  endif()

  # Link cctv only for the C ABI test; the C++ core is linked via TUs above.
  if(CCTV_TESTS_SKIP_LIBCCTV)
    target_link_libraries(cctv_tests PRIVATE GTest::gtest_main)
  else()
    target_link_libraries(cctv_tests PRIVATE cctv GTest::gtest_main)
  endif()

  if(CCTV_MODE STREQUAL "onnx")
    target_link_libraries(cctv_tests PRIVATE ${ONNXRUNTIME_LIB})
    set_source_files_properties(
      src/backend_onnx/onnx_backend.cc
      PROPERTIES COMPILE_OPTIONS "-std=c++17"
    )
  endif()

  # Suppress C++17 extension warning triggered by GTest macros expanding
  # [[maybe_unused]] in C++14 compilation mode.
  # Also suppress ORT header diagnostics.
  # Clang-only, and C++-only: the test target also compiles the generated
  # TVM kernels, which are C, and GCC rejects these flags outright.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(cctv_tests PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-c++17-extensions>
    )
  endif()

  target_compile_definitions(cctv_tests
    PRIVATE
    CCTV_BUILDING_SHARED=1
  )

  # Keep the test executable in the default build.
  set_target_properties(cctv_tests PROPERTIES EXCLUDE_FROM_ALL FALSE)

  include(GoogleTest)
  # Increase discovery timeout: ORT dylib takes ~15 s to dlopen on first use.
  gtest_discover_tests(cctv_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60)
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
# Installs the shared library and the public header so that
#   cmake --install <build-dir> --prefix <prefix>
# produces:
#   <prefix>/lib/libcctv.so  (or .dylib on macOS)
#   <prefix>/include/cctv/cctv.h
#   <prefix>/share/licenses/mep_cctv/LICENSE
install(TARGETS cctv
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES include/cctv/cctv.h
  DESTINATION include/cctv
)

install(FILES LICENSE
  DESTINATION share/licenses/mep_cctv
)
