# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# hipBLAS-LT graph sources (stub implementations used when hipblaslt not available)
target_sources(hipdnn_ep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/blas_graph.cc
)

if(NOT hipblaslt_FOUND)
  return()
endif()

# HIP language required for hipBLAS-LT
# Use DIRECTORY to propagate property to parent scope (CMake 3.18+)
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/blas_graph.cc
  DIRECTORY ${CMAKE_SOURCE_DIR}
  PROPERTIES LANGUAGE HIP
)

target_link_libraries(hipdnn_ep PRIVATE roc::hipblaslt)
target_compile_definitions(hipdnn_ep PRIVATE HIPDNN_EP_HAS_HIPBLASLT)
