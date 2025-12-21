// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/node_compute_info.h"
#include "hipdnn_ep/ep.h"
#include "hipdnn_ep/kernel.h"

namespace hipdnn_ep {

NodeComputeInfo::NodeComputeInfo(HipDNNEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

/*static*/
OrtStatus* ORT_API_CALL NodeComputeInfo::CreateStateImpl(
    OrtNodeComputeInfo* this_ptr,
    OrtNodeComputeContext* compute_context,
    void** compute_state) {

  auto* info = static_cast<NodeComputeInfo*>(this_ptr);
  HipDNNEp& ep = info->ep;

  std::string node_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
  Kernel* kernel = ep.GetKernel(node_name);
  if (kernel == nullptr) {
    RETURN_ERROR(ep.ort_api, ORT_EP_FAIL, "Kernel not found for node: " << node_name);
  }

  *compute_state = kernel;
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL NodeComputeInfo::ComputeImpl(
    OrtNodeComputeInfo* /*this_ptr*/,
    void* compute_state,
    OrtKernelContext* kernel_context) {

  auto* kernel = static_cast<Kernel*>(compute_state);
  return kernel->Execute(kernel_context);
}

/*static*/
void ORT_API_CALL NodeComputeInfo::ReleaseStateImpl(
    OrtNodeComputeInfo* /*this_ptr*/,
    void* /*compute_state*/) {
  // Kernel is owned by EP, don't delete here
}

}  // namespace hipdnn_ep
