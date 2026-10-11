# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# cmake/simd.cmake
# Detect target architecture and configure SIMD compilation.
# Sets:
#   CCTV_SIMD_SOURCES     -- list of SIMD source files to compile
#   CCTV_SCALAR_SOURCES   -- list of scalar fallback source files
#   CCTV_SIMD_FLAGS       -- compiler flags for SIMD files
#   CCTV_SIMD_ENABLED_VAL -- 1 or 0 (for CCTV_SIMD_ENABLED macro)

# ---------------------------------------------------------------------------
# Kernel selection
# ---------------------------------------------------------------------------
# Native defaults to SIMD; onnx and explicit reference builds use scalar kernels.
if(CCTV_MODE STREQUAL "native" AND NOT CCTV_SCALAR_KERNELS)
  set(CCTV_SIMD ON)
else()
  set(CCTV_SIMD OFF)
endif()

set(CCTV_SIMD_SOURCES "")
set(CCTV_SCALAR_SOURCES
  "${CMAKE_SOURCE_DIR}/src/core/image_ops_scalar.cc"
  "${CMAKE_SOURCE_DIR}/src/core/nms_scalar.cc"
)
set(CCTV_SIMD_FLAGS "")
set(CCTV_SIMD_ENABLED_VAL 0)

if(CCTV_SIMD)
  # Detect architecture.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(CCTV_SIMD_ARCH "arm64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    set(CCTV_SIMD_ARCH "x86_64")
  else()
    set(CCTV_SIMD_ARCH "unknown")
  endif()

  if(CCTV_SIMD_ARCH STREQUAL "arm64")
    # Cross-builds target Pi 4 Cortex-A72; local builds may probe the host CPU.
    if(CMAKE_CROSSCOMPILING)
      set(_cctv_default_arm_cpu "cortex-a72")
    else()
      set(_cctv_default_arm_cpu "native")
    endif()
    set(CCTV_ARM_CPU "${_cctv_default_arm_cpu}" CACHE STRING
      "-mcpu value for arm64 SIMD sources (e.g. native, cortex-a72)")
    list(APPEND CCTV_SIMD_FLAGS "-mcpu=${CCTV_ARM_CPU}")
    set(CCTV_SIMD_SOURCES
      "${CMAKE_SOURCE_DIR}/src/core/image_ops_simd.cc"
      "${CMAKE_SOURCE_DIR}/src/core/nms_simd.cc"
    )
    set(CCTV_SIMD_ENABLED_VAL 1)
    message(STATUS "CCTV SIMD: arm64 NEON (-mcpu=${CCTV_ARM_CPU})")

  elseif(CCTV_SIMD_ARCH STREQUAL "x86_64")
    # No runtime dispatch: executing this configuration requires AVX2 and FMA.
    list(APPEND CCTV_SIMD_FLAGS -mavx2 -mfma)
    set(CCTV_SIMD_SOURCES
      "${CMAKE_SOURCE_DIR}/src/core/image_ops_simd.cc"
      "${CMAKE_SOURCE_DIR}/src/core/nms_simd.cc"
    )
    set(CCTV_SIMD_ENABLED_VAL 1)
    message(STATUS "CCTV SIMD: x86_64 AVX2+FMA (-mavx2 -mfma)")

  else()
    # Report scalar selection for unsupported SIMD architectures.
    message(STATUS "CCTV SIMD: unknown arch '${CCTV_SIMD_ARCH}', using scalar")
    set(CCTV_SIMD_SOURCES ${CCTV_SCALAR_SOURCES})
  endif()

else()
  # Explicit scalar selection replaces the SIMD translation units.
  set(CCTV_SIMD_SOURCES ${CCTV_SCALAR_SOURCES})
  message(STATUS "CCTV SIMD: disabled (scalar fallback)")
endif()
