# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  add_subdirectory(hipdnn_ep/hipdnn_dialect)
  add_subdirectory(hipdnn_ep/torch_mlir_graph)
endif()
