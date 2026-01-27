# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# IRBuilder sources (stub implementations used when Torch-MLIR not available)
target_sources(hipdnn_ep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/ir_builder.cc
)

if(NOT HIPDNN_EP_ENABLE_TORCH_MLIR)
  return()
endif()

# Static library for MLIR passes (used by both hipdnn_ep and hipdnn-ep-opt)
add_library(hipdnn_ep_passes STATIC
  passes.cc
  hipdnn_offload_pass.cc
  hipdnn_graph_to_executable_pass.cc
)

target_include_directories(hipdnn_ep_passes PUBLIC
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${TORCH_MLIR_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
)

target_compile_options(hipdnn_ep_passes PRIVATE -fno-rtti)
target_compile_definitions(hipdnn_ep_passes PUBLIC HIPDNN_EP_HAS_TORCH_MLIR)

target_link_libraries(hipdnn_ep_passes PUBLIC
  hipdnn_ep_graph
  ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchOnnxToTorch.a
  ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchDialect.a
  ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchUtils.a
  MLIRFuncDialect
  MLIRCastInterfaces
  MLIRSparseTensorDialect
  MLIRDialectUtils
  MLIRTransforms
  MLIRPass
  MLIRIR
  LLVMSupport
)

add_dependencies(hipdnn_ep_passes HipDNNEpTorchMlirPassIncGen)

# Add MLIR/LLVM includes and compile flags to ir_builder
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/ir_builder.cc
  DIRECTORY ${CMAKE_SOURCE_DIR}
  PROPERTIES
    INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS};${MLIR_INCLUDE_DIRS};${TORCH_MLIR_INCLUDE_DIR};${CMAKE_BINARY_DIR}/include"
    COMPILE_OPTIONS "-fno-rtti"
)

# Link passes library to hipdnn_ep
target_link_libraries(hipdnn_ep PRIVATE hipdnn_ep_passes)
