// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"
#include <hip/hip_runtime.h>

namespace hipdnn_ep {

// Data transfer implementation for CPU <-> HIP device copies
struct HipDataTransfer : OrtDataTransferImpl, ApiPtrs {
  HipDataTransfer(ApiPtrs api_ptrs, const OrtMemoryDevice* device_mem_info, int device_id);

  static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                       const OrtMemoryDevice* src_memory_device,
                                       const OrtMemoryDevice* dst_memory_device) noexcept;

  static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                 const OrtValue** src_tensors_ptr,
                                                 OrtValue** dst_tensors_ptr,
                                                 OrtSyncStream** streams_ptr,
                                                 size_t num_tensors) noexcept;

  static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

 private:
  const OrtMemoryDevice* device_mem_info_;
  int device_id_;
};

}  // namespace hipdnn_ep
