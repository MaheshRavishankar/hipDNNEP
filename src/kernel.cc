// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/kernel.h"

namespace hipdnn_ep {

//
// Kernel implementation
//

Kernel::Kernel(const OrtApi& ort_api, const OrtLogger& logger, hipdnnHandle_t handle,
               hipblaslt_handle_t hipblaslt_handle)
    : ort_api_(ort_api), logger_(logger), handle_(handle), hipblaslt_handle_(hipblaslt_handle) {
}

Kernel::~Kernel() = default;

OrtStatus* Kernel::BuildAndCompile(Ort::ConstGraph graph) {
  try {
    // Extract graph input/output info
    std::vector<Ort::ConstValueInfo> graph_inputs = graph.GetInputs();
    std::vector<Ort::ConstValueInfo> graph_outputs = graph.GetOutputs();
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();

    // Try hipBLAS-LT first if available
    if (hipblaslt_handle_ != nullptr) {
      blas_graph_ = std::make_unique<BlasGraph>(ort_api_, hipblaslt_handle_);
      OrtStatus* status = blas_graph_->Build(graph_inputs, graph_outputs, nodes);
      if (status == nullptr) {
        return nullptr;  // Success
      }
      // BlasGraph can't handle this graph, fall back to hipDNN
      ort_api_.ReleaseStatus(status);
      blas_graph_.reset();
    }

    // Standard hipDNN graph path
    hipdnn_graph_ = std::make_unique<HipDNNGraph>(ort_api_, handle_);
    return hipdnn_graph_->Build(graph_inputs, graph_outputs, nodes);

  } catch (const std::exception& ex) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Exception building graph: " << ex.what());
  }

  return nullptr;
}

OrtStatus* Kernel::Execute(OrtKernelContext* kernel_ctx) {
  if (blas_graph_) {
    return blas_graph_->Execute(kernel_ctx);
  }
  if (hipdnn_graph_) {
    return hipdnn_graph_->Execute(kernel_ctx);
  }
  RETURN_ERROR(ort_api_, ORT_EP_FAIL, "No compiled graph available for execution");
}

}  // namespace hipdnn_ep
