# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Include subdirectory CMakeLists for modular source/config management
# (must be after add_library since they use target_sources/target_link_libraries)
add_subdirectory(core)
add_subdirectory(hipdnn_graph)
if(hipblaslt_FOUND)
  add_subdirectory(blas_graph)
endif()
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  add_subdirectory(torch_mlir_graph)
endif()
