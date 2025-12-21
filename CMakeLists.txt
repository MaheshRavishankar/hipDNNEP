# Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.20)
project(hipdnn_ep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(HIPDNN_EP_BUILD_TESTS "Build tests" ON)

# TheRock installation root - contains HIP, hipDNN, and other ROCm components
set(THEROCK_DIST "$ENV{THEROCK_DIST}" CACHE PATH "Path to TheRock dist/rocm directory")
if(NOT THEROCK_DIST)
  message(FATAL_ERROR "THEROCK_DIST must be specified. Set it via -DTHEROCK_DIST=<path> or THEROCK_DIST environment variable.")
endif()

# Add TheRock to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/lib/cmake")

# Third-party dependencies from TheRock (nlohmann-json is required by hipdnn_sdk)
list(APPEND CMAKE_PREFIX_PATH "${THEROCK_DIST}/../../third-party/nlohmann-json/dist/share/cmake")

# Find HIP from TheRock
find_package(hip REQUIRED CONFIG)

# Find hipDNN from TheRock
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend CONFIG REQUIRED)

# iree-compile is required at runtime by hipDNN backend
find_program(IREE_COMPILE iree-compile)
if(NOT IREE_COMPILE)
  message(FATAL_ERROR "iree-compile not found in PATH. It is required by hipDNN backend for code generation.")
endif()
message(STATUS "Found iree-compile: ${IREE_COMPILE}")

# ONNXRuntime headers
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNXRuntime source")
if(NOT ONNXRUNTIME_ROOT)
  message(FATAL_ERROR "ONNXRUNTIME_ROOT must be specified. Set it via -DONNXRUNTIME_ROOT=<path> or ONNXRUNTIME_ROOT environment variable.")
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")

# Verify ONNXRuntime headers exist
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "ONNXRuntime headers not found at ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

# Main library
add_library(hipdnn_ep SHARED
  src/ep_utils.cc
  src/ep_factory.cc
  src/ep.cc
  src/ep_allocator.cc
  src/ep_data_transfer.cc
  src/hipdnn_ep_exports.cc
  src/kernel.cc
  src/node_compute_info.cc
)

target_include_directories(hipdnn_ep
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(hipdnn_ep
  PRIVATE
    hipdnn_frontend
    hipdnn_backend
    hip::host
)

# Symbol visibility
target_compile_definitions(hipdnn_ep PRIVATE HIPDNN_EP_EXPORTS)

if(UNIX AND NOT APPLE)
  target_compile_options(hipdnn_ep PRIVATE -fvisibility=hidden)
  # Export only the required symbols
  set_target_properties(hipdnn_ep PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/exports.map"
  )
endif()

if(APPLE)
  target_compile_options(hipdnn_ep PRIVATE -fvisibility=hidden)
endif()

# Compile definitions
target_compile_definitions(hipdnn_ep PRIVATE
  ORT_API_MANUAL_INIT
)

# Tests
if(HIPDNN_EP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# Install
install(TARGETS hipdnn_ep
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/hipdnn_ep
  DESTINATION include
)
