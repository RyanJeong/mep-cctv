# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# cmake/optimizations.cmake
# Release-mode optimization flags, LTO, and dead-strip.
# Portable across Apple clang and GNU ld.

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR
  CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

  # -O3 + function/data sections for better dead-stripping.
  add_compile_options(-O3 -ffunction-sections -fdata-sections)

  # LTO (thin LTO on Clang, full LTO on GCC).
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "LTO: enabled")
  else()
    message(STATUS "LTO: not supported -- ${ipo_output}")
  endif()

  # Dead-strip at link time.
  if(APPLE)
    add_link_options(-Wl,-dead_strip)
    message(STATUS "Link dead-strip: -Wl,-dead_strip (Apple)")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_link_options(-Wl,--gc-sections)
    message(STATUS "Link dead-strip: -Wl,--gc-sections (Linux)")
  else()
    message(STATUS "Link dead-strip: skipped (unsupported platform)")
  endif()

endif()
