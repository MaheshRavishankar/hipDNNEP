# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Torch-MLIR lit testing infrastructure

# Find ONNXRuntime library for the tool (may already be found by parent)
if(NOT ONNXRUNTIME_LIB)
  set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
  find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_LIB_DIR}
  )
endif()

if(NOT ONNXRUNTIME_LIB)
  message(WARNING "ONNXRuntime library not found - lit testing tool will not be built")
  return()
endif()

# Build the hipdnn-onnx-to-torch-mlir tool
add_executable(hipdnn-onnx-to-torch-mlir
  ${CMAKE_SOURCE_DIR}/tools/hipdnn-onnx-to-torch-mlir.cc
)
target_include_directories(hipdnn-onnx-to-torch-mlir PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
)
target_compile_definitions(hipdnn-onnx-to-torch-mlir PRIVATE
  ORT_API_MANUAL_INIT
  HIPDNN_EP_LIB_PATH="$<TARGET_FILE:hipdnn_ep>"
)
target_link_libraries(hipdnn-onnx-to-torch-mlir PRIVATE
  ${ONNXRUNTIME_LIB}
  dl
)
add_dependencies(hipdnn-onnx-to-torch-mlir hipdnn_ep)

# Lit testing setup
set(FILECHECK_PATH "${TORCH_MLIR_INSTALL_DIR}/bin/FileCheck")
if(NOT EXISTS "${FILECHECK_PATH}")
  message(STATUS "FileCheck not found at ${FILECHECK_PATH} - lit tests will not be available")
  return()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

find_program(LIT_COMMAND
  NAMES lit llvm-lit
  HINTS
    ${TORCH_MLIR_INSTALL_DIR}/bin
    ${TORCH_MLIR_INSTALL_DIR}/../llvm-build/bin
  DOC "Path to lit test runner"
)

if(NOT LIT_COMMAND)
  message(STATUS "lit not found - lit tests will not be available")
  return()
endif()

add_custom_target(check-torch-mlir
  COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR} -v
  DEPENDS hipdnn-onnx-to-torch-mlir
  COMMENT "Running Torch-MLIR lit tests"
)

add_test(NAME torch-mlir-lit-tests
  COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}
)
