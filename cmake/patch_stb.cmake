# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# Keep the pinned upstream source intact and patch only the build-local header.
# JPEG bit accumulation must use unsigned shifts (stb's signed buffer is UB).
file(READ "${stb_SOURCE_DIR}/stb_image_write.h" _stb_header)
set(_stb_old
  "int *bitBufP, int *bitCntP"
  "int bitBuf = *bitBufP, bitCnt = *bitCntP\;"
  "bs[0] << (24 - bitCnt)"
  "int *bitBuf, int *bitCnt"
  "int bitBuf=0, bitCnt=0\;"
)
set(_stb_new
  "unsigned int *bitBufP, int *bitCntP"
  "unsigned int bitBuf = *bitBufP\; int bitCnt = *bitCntP\;"
  "(unsigned int)bs[0] << (24 - bitCnt)"
  "unsigned int *bitBuf, int *bitCnt"
  "unsigned int bitBuf=0\; int bitCnt=0\;"
)
foreach(_old _new IN ZIP_LISTS _stb_old _stb_new)
  string(FIND "${_stb_header}" "${_old}" _stb_match)
  if(_stb_match EQUAL -1)
    message(FATAL_ERROR "Pinned stb JPEG patch no longer applies: ${_old}")
  endif()
  string(REPLACE "${_old}" "${_new}" _stb_header "${_stb_header}")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_deps/stb-patched/stb_image_write.h"
  "${_stb_header}")
