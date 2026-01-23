# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  add_subdirectory(hipdnn_ep_opt)
  add_subdirectory(hipdnn_onnx_to_torch_mlir)
endif()
