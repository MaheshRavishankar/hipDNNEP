# Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.20)
project(hipdnn_ep LANGUAGES C CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(HIPDNN_EP_BUILD_TESTS "Build tests" ON)
option(HIPDNN_EP_ENABLE_TORCH_MLIR "Enable Torch-MLIR based IR pipeline" OFF)

# Torch-MLIR configuration
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  set(TORCH_MLIR_INSTALL_DIR "" CACHE PATH "Path to torch-mlir installation (contains lib/cmake/torch-mlir)")
  if(NOT TORCH_MLIR_INSTALL_DIR)
    message(FATAL_ERROR "TORCH_MLIR_INSTALL_DIR must be specified when HIPDNN_EP_ENABLE_TORCH_MLIR is ON")
  endif()
endif()

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

# Find hipBLAS-LT from TheRock (optional)
find_package(hipblaslt CONFIG)
if(hipblaslt_FOUND)
  message(STATUS "Found hipBLAS-LT: MatMul/Gemm support enabled")
else()
  message(STATUS "hipBLAS-LT not found: MatMul/Gemm support disabled")
endif()

# Find Torch-MLIR and MLIR (optional)
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  # Add torch-mlir install directory to CMAKE_PREFIX_PATH
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_MLIR_INSTALL_DIR}/lib/cmake/mlir")
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_MLIR_INSTALL_DIR}/lib/cmake/llvm")

  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Found MLIR: ${MLIR_DIR}")
  message(STATUS "Found LLVM: ${LLVM_DIR}")

  # MLIR/LLVM CMake modules
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  # Torch-MLIR libraries (no CMake config file, so we find them directly)
  set(TORCH_MLIR_LIB_DIR "${TORCH_MLIR_INSTALL_DIR}/lib")
  set(TORCH_MLIR_INCLUDE_DIR "${TORCH_MLIR_INSTALL_DIR}/include")
endif()

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

# Main library sources
set(HIPDNN_EP_SOURCES
  src/ep_utils.cc
  src/ep_factory.cc
  src/ep.cc
  src/ep_allocator.cc
  src/ep_data_transfer.cc
  src/hipdnn_ep_exports.cc
  src/hipdnn_graph.cc
  src/kernel.cc
  src/node_compute_info.cc
)

# Add hipBLAS-LT sources if available (compiled with HIP/hipcc)
if(hipblaslt_FOUND)
  list(APPEND HIPDNN_EP_SOURCES src/blas_graph.cc)
  set_source_files_properties(src/blas_graph.cc PROPERTIES LANGUAGE HIP)
endif()

# Add Torch-MLIR sources if enabled
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  list(APPEND HIPDNN_EP_SOURCES src/ir_builder.cc)
endif()

add_library(hipdnn_ep SHARED ${HIPDNN_EP_SOURCES})

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

# Link hipBLAS-LT if available
if(hipblaslt_FOUND)
  target_link_libraries(hipdnn_ep PRIVATE roc::hipblaslt)
  target_compile_definitions(hipdnn_ep PRIVATE HIPDNN_EP_HAS_HIPBLASLT)
endif()

# Torch-MLIR configuration
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  target_compile_definitions(hipdnn_ep PRIVATE HIPDNN_EP_HAS_TORCH_MLIR)

  # Add MLIR/LLVM includes only to ir_builder.cc
  set_source_files_properties(src/ir_builder.cc PROPERTIES
    INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS};${MLIR_INCLUDE_DIRS};${TORCH_MLIR_INCLUDE_DIR}"
  )

  # Link Torch-MLIR and MLIR libraries
  # Note: TorchMLIRTorchDialect has transitive dependencies on TorchUtils,
  # CastInterfaces, SparseTensorDialect, and DialectUtils
  target_link_libraries(hipdnn_ep PRIVATE
    ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchDialect.a
    ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchUtils.a
    MLIRFuncDialect
    MLIRCastInterfaces
    MLIRSparseTensorDialect
    MLIRDialectUtils
    MLIRIR
    LLVMSupport
  )
endif()

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

# Tools for testing (only when Torch-MLIR is enabled)
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  # Find ONNXRuntime library for the tool
  set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/build/RelWithDebInfo" CACHE PATH "ONNXRuntime library directory")
  find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_LIB_DIR}
  )

  if(ONNXRUNTIME_LIB)
    add_executable(hipdnn-onnx-to-torch-mlir tools/hipdnn-onnx-to-torch-mlir.cc)
    target_include_directories(hipdnn-onnx-to-torch-mlir PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
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

    # Lit testing infrastructure
    set(FILECHECK_PATH "${TORCH_MLIR_INSTALL_DIR}/bin/FileCheck")
    if(EXISTS "${FILECHECK_PATH}")
      configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/test/lit/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/test/lit/lit.site.cfg.py
        @ONLY
      )

      find_program(LIT_COMMAND
        NAMES lit llvm-lit
        HINTS
          ${TORCH_MLIR_INSTALL_DIR}/bin
          ${TORCH_MLIR_INSTALL_DIR}/../llvm-build/bin
        DOC "Path to lit test runner"
      )

      if(LIT_COMMAND)
        add_custom_target(check-torch-mlir
          COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/test/lit -v
          DEPENDS hipdnn-onnx-to-torch-mlir
          COMMENT "Running Torch-MLIR lit tests"
        )

        add_test(NAME torch-mlir-lit-tests
          COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/test/lit
        )
      else()
        message(STATUS "lit not found - lit tests will not be available")
      endif()
    else()
      message(STATUS "FileCheck not found at ${FILECHECK_PATH} - lit tests will not be available")
    endif()
  else()
    message(WARNING "ONNXRuntime library not found at ${ONNXRUNTIME_LIB_DIR} - lit testing tool will not be built")
  endif()
endif()

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
