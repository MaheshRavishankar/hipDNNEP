# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Find Torch-MLIR and MLIR
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

# Torch-MLIR IR builder sources
target_sources(hipdnn_ep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/ir_builder.cc
)

# Add MLIR/LLVM includes only to ir_builder.cc
# Use DIRECTORY to propagate property to parent scope (CMake 3.18+)
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/ir_builder.cc
  DIRECTORY ${CMAKE_SOURCE_DIR}
  PROPERTIES
    INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS};${MLIR_INCLUDE_DIRS};${TORCH_MLIR_INCLUDE_DIR}"
)

target_compile_definitions(hipdnn_ep PRIVATE HIPDNN_EP_HAS_TORCH_MLIR)

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
