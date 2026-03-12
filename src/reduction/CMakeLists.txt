# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for GPU reduction operations
set(LIB_NAME hipdnn_ep_src_reduction)
set(REDUCTION_SOURCES
  reduction_graph.cc
)

add_library(${LIB_NAME} STATIC ${REDUCTION_SOURCES})
add_library(hipdnn_ep::src::reduction ALIAS ${LIB_NAME})

# HIP kernel source — compiled with the HIP compiler
set_source_files_properties(${REDUCTION_SOURCES} PROPERTIES LANGUAGE HIP)

target_link_libraries(${LIB_NAME} PUBLIC
  hipdnn_ep::src::utils
  hip::host
)
