# Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Find Python for generating test models
# Use Python3_EXECUTABLE if set, otherwise default to local .venv
if(NOT Python3_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
  set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Find GTest
find_package(GTest CONFIG)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

# ONNXRuntime library for testing
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")

find_library(ONNXRUNTIME_LIB onnxruntime
  HINTS ${ONNXRUNTIME_LIB_DIR}
)

if(NOT ONNXRUNTIME_LIB)
  message(WARNING "ONNXRuntime library not found at ${ONNXRUNTIME_LIB_DIR}. Tests will not be built.")
  return()
endif()

# Test executable (sources added via subdirectories)
add_executable(hipdnn_ep_tests)

target_include_directories(hipdnn_ep_tests PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(hipdnn_ep_tests PRIVATE
  GTest::gtest
  GTest::gtest_main
  ${ONNXRUNTIME_LIB}
  hip::host
  dl
)

target_compile_definitions(hipdnn_ep_tests PRIVATE
  HIPDNN_EP_LIB_PATH="$<TARGET_FILE:hipdnn_ep>"
  ORT_API_MANUAL_INIT
)

# Include test subdirectories (add sources, models, and compile definitions)
add_subdirectory(core)
add_subdirectory(hipdnn)
add_subdirectory(hipblaslt)
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  add_subdirectory(torch_mlir)
endif()

# Add test discovery
include(GoogleTest)
gtest_discover_tests(hipdnn_ep_tests)
