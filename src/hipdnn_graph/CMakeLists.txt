# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for HipDNNGraph
set(LIB_NAME hipdnn_ep_src_hipdnn_graph)
set(HIPDNN_GRAPH_SOURCES
  hipdnn_graph.cc
)

add_library(${LIB_NAME} STATIC ${HIPDNN_GRAPH_SOURCES})
add_library(hipdnn_ep::src::hipdnn_graph ALIAS ${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_include_directories(${LIB_NAME} SYSTEM PUBLIC
  ${HIPDNN_FRONTEND_INCLUDE_DIR}
  ${HIPDNN_SDK_INCLUDE_DIR}
  ${HIPDNN_BACKEND_INCLUDE_DIR}
  ${NLOHMANN_JSON_INCLUDE_DIR}
  ${ROCM_INCLUDE_DIR}
)

target_link_libraries(${LIB_NAME} PUBLIC
  hipdnn_ep::src::utils
  hipdnn_frontend
  hipdnn_backend
  hip::host
)

# Add MLIR includes when Torch-MLIR is enabled (for MLIR-based Build method)
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  target_include_directories(${LIB_NAME} PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${TORCH_MLIR_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
  )
  target_compile_options(${LIB_NAME} PRIVATE -fno-rtti)
  target_compile_definitions(${LIB_NAME} PUBLIC HIPDNN_EP_HAS_TORCH_MLIR)
  target_link_libraries(${LIB_NAME} PUBLIC
    ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchDialect.a
    MLIRIR
    LLVMSupport
  )
endif()
