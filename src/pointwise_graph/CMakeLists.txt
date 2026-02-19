# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for pointwise operations via HIP kernels
set(LIB_NAME hipdnn_ep_src_pointwise_graph)
set(POINTWISE_GRAPH_SOURCES
  pointwise_graph.cc
)

set_source_files_properties(${POINTWISE_GRAPH_SOURCES} PROPERTIES LANGUAGE HIP)

add_library(${LIB_NAME} STATIC ${POINTWISE_GRAPH_SOURCES})
add_library(hipdnn_ep::src::pointwise_graph ALIAS ${LIB_NAME})

target_link_libraries(${LIB_NAME} PUBLIC
  hipdnn_ep::src::utils
  hip::host
)
