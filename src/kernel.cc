// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/kernel.h"

#include <iostream>

namespace hipdnn_ep {

//
// Kernel implementation
//

Kernel::Kernel(const OrtApi& ort_api, const OrtLogger& logger, KernelConfig&& config)
    : ort_api_(ort_api), logger_(logger), config_(std::move(config)) {}

Kernel::~Kernel() = default;

OrtStatus* Kernel::BuildAndCompile(Ort::ConstGraph graph) {
  try {
    // Extract graph input/output info
    std::vector<Ort::ConstValueInfo> graph_inputs = graph.GetInputs();
    std::vector<Ort::ConstValueInfo> graph_outputs = graph.GetOutputs();
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();

    // Use Torch-MLIR path if requested
    if (config_.useTorchMlir()) {
      ir_builder_ = std::make_unique<IRBuilder>();
      if (!ir_builder_->BuildModule(graph_inputs, graph_outputs, nodes)) {
        RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Failed to build Torch-MLIR module");
      }
      // TODO: Lower and compile the MLIR module
      if (config_.dumpTorchMlir()) {
        // Print to stdout for lit testing
        std::cout << ir_builder_->PrintModule() << std::flush;
      } else {
        LOG(ort_api_, logger_, INFO, "Generated Torch-MLIR:\n"
                                         << ir_builder_->PrintModule());
      }
      return nullptr;
    }

    // Try hipBLAS-LT first if available
    if (config_.useHipBlasLT()) {
      blas_graph_ = std::make_unique<BlasGraph>(ort_api_, config_.getHipBlasLtHandle());
      OrtStatus* status = blas_graph_->Build(graph_inputs, graph_outputs, nodes);
      if (status == nullptr) {
        return nullptr;  // Success
      }
      // BlasGraph can't handle this graph, fall back to hipDNN
      ort_api_.ReleaseStatus(status);
      blas_graph_.reset();
    }

    // Standard hipDNN graph path
    if (config_.useHipDNN()) {
      hipdnn_graph_ = std::make_unique<HipDNNGraph>(ort_api_, config_.getHipDNNHandle());
      return hipdnn_graph_->Build(graph_inputs, graph_outputs, nodes);
    }

    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Unable to build and compile graph");

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
