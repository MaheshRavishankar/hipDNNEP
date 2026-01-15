// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"
#include <hip/hip_runtime.h>

namespace hipdnn_ep {

class HipDNNEpFactory;

/// @brief Memcpy kernel implementation for MemcpyToHost and MemcpyFromHost operations
/// This kernel handles data transfers between CPU and GPU memory
struct MemcpyKernelImpl : OrtKernelImpl {
  enum class Direction {
    ToHost,    // GPU -> CPU (MemcpyToHost)
    FromHost   // CPU -> GPU (MemcpyFromHost)
  };

  MemcpyKernelImpl(HipDNNEpFactory& factory, Direction direction, int device_id);

  static OrtStatus* ORT_API_CALL ComputeImpl(
      OrtKernelImpl* this_ptr,
      OrtKernelContext* context) noexcept;

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  HipDNNEpFactory& factory_;
  Direction direction_;
  int device_id_;
};

/// @brief Creates a MemcpyToHost kernel
/// @param kernel_create_func_state Pointer to HipDNNEpFactory
/// @param info Kernel info
/// @param kernel_out Output kernel
OrtStatus* ORT_API_CALL CreateMemcpyToHostKernel(
    void* kernel_create_func_state,
    const OrtKernelInfo* info,
    OrtKernelImpl** kernel_out);

/// @brief Creates a MemcpyFromHost kernel
/// @param kernel_create_func_state Pointer to HipDNNEpFactory
/// @param info Kernel info
/// @param kernel_out Output kernel
OrtStatus* ORT_API_CALL CreateMemcpyFromHostKernel(
    void* kernel_create_func_state,
    const OrtKernelInfo* info,
    OrtKernelImpl** kernel_out);

/// @brief Register MemcpyToHost and MemcpyFromHost kernels in the kernel registry
/// @param factory The EP factory
/// @param kernel_registry The kernel registry to add kernels to
/// @param ep_name The execution provider name
OrtStatus* RegisterMemcpyKernels(
    HipDNNEpFactory& factory,
    OrtKernelRegistry* kernel_registry,
    const char* ep_name);

}  // namespace hipdnn_ep
