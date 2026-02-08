# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for hipBLAS-LT graph (stub when hipblaslt not available)
set(LIB_NAME hipdnn_ep_src_blas_graph)
set(BLAS_GRAPH_SOURCES
  blas_graph.cc
)

add_library(${LIB_NAME} STATIC ${BLAS_GRAPH_SOURCES})
add_library(hipdnn_ep::src::blas_graph ALIAS ${LIB_NAME})

target_link_libraries(${LIB_NAME} PUBLIC
  hipdnn_ep::src::utils
)

if(NOT hipblaslt_FOUND)
  return()
endif()

set_source_files_properties(${BLAS_GRAPH_SOURCES} PROPERTIES LANGUAGE HIP)
target_link_libraries(${LIB_NAME} PUBLIC roc::hipblaslt)
target_compile_definitions(${LIB_NAME} PUBLIC HIPDNN_EP_HAS_HIPBLASLT)
