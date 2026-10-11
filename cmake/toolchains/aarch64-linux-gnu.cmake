# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# cmake/toolchains/aarch64-linux-gnu.cmake
# Cross-compile for 64-bit Raspberry Pi OS (Cortex-A72 / Raspberry Pi 4).
#
# Usage:
#   cmake -B build-pi \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake \
#     -DCMAKE_BUILD_TYPE=Release -DCCTV_MODE=native
#
# Requires the Debian/Ubuntu cross toolchain:
#   apt-get install g++-aarch64-linux-gnu
#
# No separate sysroot is set: the Debian cross packages ship their own headers
# and libraries under /usr/aarch64-linux-gnu, which the compiler finds by
# default.  Set CMAKE_SYSROOT below only if you are linking against a real
# Raspberry Pi OS image instead.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Where to look for target libraries. Debian multiarch installs foreign-arch
# packages (e.g. libasound2-dev:arm64) under /usr/lib/aarch64-linux-gnu, so
# that path must be searchable or find_library(asound) misses it and the
# microphone backend is silently dropped from the build.
set(CMAKE_FIND_ROOT_PATH
  /usr/aarch64-linux-gnu
  /usr/lib/aarch64-linux-gnu
  /usr/include/aarch64-linux-gnu)
list(APPEND CMAKE_LIBRARY_ARCHITECTURE aarch64-linux-gnu)

# Search host paths for programs, but only target paths for headers/libraries,
# so the linker never picks up an x86_64 .so by accident.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BEFORE)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
