# Copyright (c) 2026 Munseong Jeong <ryan.m.jeong@hotmail.com>

# cmake/toolchains/x86_64-linux-gnu.cmake
# Cross-compile for 64-bit x86 Linux (lab desktops, CI runners) from a
# non-x86_64 host -- typically an Apple Silicon or arm64 Linux machine.
#
# Usage:
#   cmake -B build-x86 \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/x86_64-linux-gnu.cmake \
#     -DCMAKE_BUILD_TYPE=Release -DCCTV_MODE=native
#
# Requires the Debian/Ubuntu cross toolchain:
#   apt-get install crossbuild-essential-amd64
#
# The Pi 4 target lives in aarch64-linux-gnu.cmake; this file exists so the
# same Docker image can also produce the x86_64 build students run on their
# own laptops.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(CMAKE_C_COMPILER x86_64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER x86_64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BEFORE)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
