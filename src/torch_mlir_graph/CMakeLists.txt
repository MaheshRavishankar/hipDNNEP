# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for Torch-MLIR graph (stub when Torch-MLIR not available)
set(LIB_NAME hipdnn_ep_src_torch_mlir_graph)
set(TORCH_MLIR_GRAPH_SOURCES ir_builder.cc)

if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  list(APPEND TORCH_MLIR_GRAPH_SOURCES
    passes.cc
    hipdnn_offload_pass.cc
    hipdnn_graph_to_executable_pass.cc
    hipdnn_backend_legalize_pass.cc
    hipdnn_finalize_memrefs_pass.cc
  )
endif()

add_library(${LIB_NAME} STATIC ${TORCH_MLIR_GRAPH_SOURCES})
add_library(hipdnn_ep::src::torch_mlir_graph ALIAS ${LIB_NAME})

target_link_libraries(${LIB_NAME} PUBLIC hipdnn_ep::src::utils)

if(NOT HIPDNN_EP_ENABLE_TORCH_MLIR)
  return()
endif()

target_include_directories(${LIB_NAME} PUBLIC
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${TORCH_MLIR_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
)

target_compile_options(${LIB_NAME} PRIVATE -fno-rtti)
target_compile_definitions(${LIB_NAME} PUBLIC HIPDNN_EP_HAS_TORCH_MLIR)

target_link_libraries(${LIB_NAME} PUBLIC
  hipdnn_ep::src::hipdnn_graph
  hipdnn_ep::src::hipdnn_dialect
  ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchOnnxToTorch.a
  ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchDialect.a
  ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchUtils.a
  ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchConversionDialect.a
  ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchConversionPasses.a
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRCastInterfaces
  MLIRSparseTensorDialect
  MLIRDialectUtils
  MLIRTransforms
  MLIRPass
  MLIRIR
  LLVMSupport
)

add_dependencies(${LIB_NAME} HipDNNEpTorchMlirPassIncGen)
