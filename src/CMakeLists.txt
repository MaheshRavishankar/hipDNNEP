# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Each subdirectory creates a static library that is linked into hipdnn_ep
add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(hipdnn_graph)
add_subdirectory(blas_graph)
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  add_subdirectory(hipdnn_dialect)
endif()
add_subdirectory(torch_mlir_graph)
