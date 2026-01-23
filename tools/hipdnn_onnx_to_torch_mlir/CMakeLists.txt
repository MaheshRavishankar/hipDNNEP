# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Find ONNXRuntime library for the tool
if(NOT ONNXRUNTIME_LIB)
  set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
  find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_LIB_DIR}
  )
endif()

if(NOT ONNXRUNTIME_LIB)
  message(STATUS "ONNXRuntime library not found - hipdnn-onnx-to-torch-mlir will not be built")
  return()
endif()

add_executable(hipdnn-onnx-to-torch-mlir
  hipdnn-onnx-to-torch-mlir.cc
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
