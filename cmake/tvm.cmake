# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# cmake/tvm.cmake
# Locate the Apache TVM runtime needed by the TVM-compiled native backend.
#
# Included only when native mode selects a real generated model.
# An onnx build must never see any of this, and a native build must never see
# ONNX Runtime -- the two backends are mutually exclusive.
#
# Two ways to supply the runtime, in order of preference:
#
#   1. STATIC (-DCCTV_TVM_STATIC_ROOT=<dir> containing libtvm_runtime.a and
#      libtvm_ffi_static.a). Everything ends up inside libcctv, so the
#      deployed artifact is a single shared library with no TVM .so beside
#      it. This is what the Raspberry Pi image builds.
#
#   2. SHARED (-DCCTV_TVM_ROOT=<site-packages> from the apache-tvm wheel).
#      Convenient for local experiments, but it means shipping
#      libtvm_runtime.so alongside libcctv.
#
# Sets:
#   CCTV_TVM_INCLUDE_DIRS  headers for tvm_ffi + dlpack
#   CCTV_TVM_LIBRARIES     link arguments, whole-archive wrapped when static

set(CCTV_TVM_STATIC_ROOT "" CACHE PATH
  "Directory containing libtvm_runtime.a and libtvm_ffi_static.a")
set(CCTV_TVM_ROOT "" CACHE PATH
  "site-packages directory containing the tvm/ and tvm_ffi/ packages")
set(CCTV_TVM_INCLUDE_ROOT "" CACHE PATH
  "Directory containing tvm/ and tvm/ffi/ headers (defaults to CCTV_TVM_ROOT)")

# ---------------------------------------------------------------------------
# Static: preferred, and the only option that keeps one shared library
# ---------------------------------------------------------------------------
if(CCTV_TVM_STATIC_ROOT)
  set(_cctv_static_runtime "${CCTV_TVM_STATIC_ROOT}/libtvm_runtime.a")
  set(_cctv_static_ffi "${CCTV_TVM_STATIC_ROOT}/libtvm_ffi_static.a")
  foreach(_lib "${_cctv_static_runtime}" "${_cctv_static_ffi}")
    if(NOT EXISTS "${_lib}")
      message(FATAL_ERROR "static TVM runtime not found: ${_lib}")
    endif()
  endforeach()

  # Preserve static registration objects with a platform-aware whole-archive link.
  # See models/README.md, Appendix: conversion and linking decisions.
  set(_cctv_static_libs "${_cctv_static_runtime}" "${_cctv_static_ffi}")
  if(EXISTS "${CCTV_TVM_STATIC_ROOT}/libbacktrace.a")
    list(APPEND _cctv_static_libs "${CCTV_TVM_STATIC_ROOT}/libbacktrace.a")
  endif()
  set(CCTV_TVM_LIBRARIES
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_cctv_static_libs}>")

  set(CCTV_TVM_LINK_KIND "static (linked into libcctv)")

  # ---------------------------------------------------------------------------
  # Shared: fallback for local experiments
  # ---------------------------------------------------------------------------
else()
  if(NOT CCTV_TVM_ROOT)
    file(GLOB _cctv_tvm_candidates
      "${CMAKE_SOURCE_DIR}/.venv-convert/lib/python3*/site-packages")
    if(_cctv_tvm_candidates)
      list(GET _cctv_tvm_candidates 0 CCTV_TVM_ROOT)
      message(STATUS "TVM: using in-tree venv ${CCTV_TVM_ROOT}")
    endif()
  endif()
  if(NOT CCTV_TVM_ROOT)
    message(FATAL_ERROR
      "CCTV_USE_GENERATED_MODEL=ON needs the TVM runtime. Pass "
      "-DCCTV_TVM_STATIC_ROOT=<dir with libtvm_runtime.a> (preferred), or "
      "-DCCTV_TVM_ROOT=<site-packages> to link the shared runtime.")
  endif()

  set(_cctv_shared_runtime "")
  set(_cctv_shared_ffi "")
  foreach(_ext so dylib)
    if(NOT _cctv_shared_runtime AND
      EXISTS "${CCTV_TVM_ROOT}/tvm/lib/libtvm_runtime.${_ext}")
      set(_cctv_shared_runtime "${CCTV_TVM_ROOT}/tvm/lib/libtvm_runtime.${_ext}")
    endif()
    if(NOT _cctv_shared_ffi AND
      EXISTS "${CCTV_TVM_ROOT}/tvm_ffi/lib/libtvm_ffi.${_ext}")
      set(_cctv_shared_ffi "${CCTV_TVM_ROOT}/tvm_ffi/lib/libtvm_ffi.${_ext}")
    endif()
  endforeach()
  if(NOT _cctv_shared_runtime OR NOT _cctv_shared_ffi)
    message(FATAL_ERROR
      "TVM shared runtime not found under ${CCTV_TVM_ROOT}.")
  endif()
  set(CCTV_TVM_LIBRARIES "${_cctv_shared_runtime}" "${_cctv_shared_ffi}")
  set(CCTV_TVM_LINK_KIND "shared (libtvm_runtime must ship alongside)")
endif()

# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
# src/backend_native/model_tvm.cc stays inside the tvm::ffi namespace: the wheel's
# copy of tvm/runtime/tensor.h includes tvm/support/io.h, which the wheel does
# not ship, so only the tvm/ffi/ headers are safe to include.
if(NOT CCTV_TVM_INCLUDE_ROOT)
  set(CCTV_TVM_INCLUDE_ROOT "${CCTV_TVM_ROOT}")
endif()

set(CCTV_TVM_INCLUDE_DIRS "")
foreach(_dir
  "${CCTV_TVM_INCLUDE_ROOT}/tvm_ffi/include"
  "${CCTV_TVM_INCLUDE_ROOT}/tvm_ffi/3rdparty/dlpack/include"
  "${CCTV_TVM_INCLUDE_ROOT}/tvm/include"
  "${CCTV_TVM_INCLUDE_ROOT}/include"
  "${CCTV_TVM_INCLUDE_ROOT}/3rdparty/tvm-ffi/include"
  "${CCTV_TVM_INCLUDE_ROOT}/3rdparty/tvm-ffi/3rdparty/dlpack/include")
  if(EXISTS "${_dir}")
    list(APPEND CCTV_TVM_INCLUDE_DIRS "${_dir}")
  endif()
endforeach()

if(NOT CCTV_TVM_INCLUDE_DIRS)
  message(FATAL_ERROR
    "No TVM headers found under ${CCTV_TVM_INCLUDE_ROOT}. Pass "
    "-DCCTV_TVM_INCLUDE_ROOT=<dir containing tvm_ffi/include>.")
endif()

message(STATUS "TVM runtime: ${CCTV_TVM_LINK_KIND}")
