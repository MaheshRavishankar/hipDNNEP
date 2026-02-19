# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for core EP implementation
set(LIB_NAME hipdnn_ep_src_core)
set(CORE_SOURCES
  ep.cc
  ep_allocator.cc
  ep_data_transfer.cc
  ep_factory.cc
  hipdnn_ep_exports.cc
  kernel.cc
  node_compute_info.cc
)

add_library(${LIB_NAME} STATIC ${CORE_SOURCES})
add_library(hipdnn_ep::src::core ALIAS ${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_compile_definitions(${LIB_NAME} PRIVATE
  HIPDNN_EP_EXPORTS
  ORT_API_MANUAL_INIT
)

target_link_libraries(${LIB_NAME} PUBLIC
  hipdnn_ep::src::utils
  hipdnn_ep::src::hipdnn_graph
  hipdnn_ep::src::blas_graph
  hipdnn_ep::src::pointwise_graph
  hipdnn_ep::src::torch_mlir_graph
)
