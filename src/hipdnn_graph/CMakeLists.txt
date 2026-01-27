# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for HipDNNGraph (used by hipdnn_ep and hipdnn_ep_passes)
add_library(hipdnn_ep_graph STATIC
  hipdnn_graph.cc
)

target_include_directories(hipdnn_ep_graph PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_include_directories(hipdnn_ep_graph SYSTEM PUBLIC
  ${HIPDNN_FRONTEND_INCLUDE_DIR}
  ${HIPDNN_SDK_INCLUDE_DIR}
  ${HIPDNN_BACKEND_INCLUDE_DIR}
  ${NLOHMANN_JSON_INCLUDE_DIR}
  ${ROCM_INCLUDE_DIR}
)

target_link_libraries(hipdnn_ep_graph PUBLIC
  hipdnn_ep_utils
  hipdnn_frontend
  hip::host
)

# Add MLIR includes when Torch-MLIR is enabled (for MLIR-based Build method)
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  target_include_directories(hipdnn_ep_graph PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${TORCH_MLIR_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
  )
  target_compile_options(hipdnn_ep_graph PRIVATE -fno-rtti)
  target_compile_definitions(hipdnn_ep_graph PUBLIC HIPDNN_EP_HAS_TORCH_MLIR)
  target_link_libraries(hipdnn_ep_graph PUBLIC
    ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchDialect.a
    MLIRIR
    LLVMSupport
  )
endif()

# Link hipdnn_ep_graph to hipdnn_ep
target_link_libraries(hipdnn_ep PRIVATE hipdnn_ep_graph)
