# Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

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

# Test executable
add_executable(hipdnn_ep_tests
  test_ep_load.cc
  test_conv.cc
  test_matmul.cc
)

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

# Copy test models to build directory
set(CONV_TEST_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/conv_test.onnx")
if(EXISTS "${CONV_TEST_MODEL}")
  configure_file("${CONV_TEST_MODEL}" "${CMAKE_CURRENT_BINARY_DIR}/conv_test.onnx" COPYONLY)
endif()

# MatMul/Gemm test models
set(MATMUL_TEST_MODELS
  matmul_test.onnx
  gemm_test.onnx
  gemm_bias_test.onnx
  gemm_trans_a_test.onnx
  gemm_trans_b_test.onnx
  gemm_scaled_test.onnx
)

foreach(MODEL ${MATMUL_TEST_MODELS})
  set(MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${MODEL}")
  if(EXISTS "${MODEL_PATH}")
    configure_file("${MODEL_PATH}" "${CMAKE_CURRENT_BINARY_DIR}/${MODEL}" COPYONLY)
  endif()
endforeach()

target_compile_definitions(hipdnn_ep_tests PRIVATE
  HIPDNN_EP_LIB_PATH="$<TARGET_FILE:hipdnn_ep>"
  CONV_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/conv_test.onnx"
  MATMUL_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/matmul_test.onnx"
  GEMM_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_test.onnx"
  GEMM_BIAS_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_bias_test.onnx"
  GEMM_TRANS_A_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_trans_a_test.onnx"
  GEMM_TRANS_B_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_trans_b_test.onnx"
  GEMM_SCALED_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_scaled_test.onnx"
  ORT_API_MANUAL_INIT
)

# Add test
include(GoogleTest)
gtest_discover_tests(hipdnn_ep_tests)
