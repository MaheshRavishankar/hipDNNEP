// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"

namespace hipdnn_ep {

class HipDNNEp;

/// @brief Node compute info for all operations
struct NodeComputeInfo : OrtNodeComputeInfo {
  explicit NodeComputeInfo(HipDNNEp& ep);

  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo* this_ptr,
      OrtNodeComputeContext* compute_context,
      void** compute_state);

  static OrtStatus* ORT_API_CALL ComputeImpl(
      OrtNodeComputeInfo* this_ptr,
      void* compute_state,
      OrtKernelContext* kernel_context);

  static void ORT_API_CALL ReleaseStateImpl(
      OrtNodeComputeInfo* this_ptr,
      void* compute_state);

  HipDNNEp& ep;
};

}  // namespace hipdnn_ep
