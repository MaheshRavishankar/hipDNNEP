# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for EP utilities (used by hipdnn_ep, hipdnn_ep_graph, etc.)
add_library(hipdnn_ep_utils STATIC
  ep_utils.cc
)

target_include_directories(hipdnn_ep_utils PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR}
)
