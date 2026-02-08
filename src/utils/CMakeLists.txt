# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for EP utilities (used by hipdnn_ep, hipdnn_ep_graph, etc.)
set(LIB_NAME hipdnn_ep_src_utils)
set(UTILS_SOURCES
  ep_utils.cc
)

add_library(${LIB_NAME} STATIC ${UTILS_SOURCES})
add_library(hipdnn_ep::src::utils ALIAS ${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR}
)
