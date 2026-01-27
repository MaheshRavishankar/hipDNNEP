// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "hipdnn_ep/core/ort_api.h"
#include "hipdnn_ep/hipdnn_graph/ep_status.h"

#include <memory>
#include <vector>

// Forward declarations for hipDNN (must match hipdnn_backend.h)
struct hipdnnHandle;
typedef hipdnnHandle* hipdnnHandle_t;

#ifdef HIPDNN_EP_HAS_TORCH_MLIR
namespace mlir {
class Region;
}  // namespace mlir
#endif

namespace hipdnn_ep {

// Forward declaration for pimpl
struct HipDNNGraphImpl;

/// @brief Encapsulates a hipDNN execution graph for convolution and other operations
///
/// This class mirrors BlasGraph's interface, providing Build and Execute methods.
/// All hipDNN-specific types are hidden in the implementation.
class HipDNNGraph {
 public:
  /// @param handle hipDNN handle
  explicit HipDNNGraph(hipdnnHandle_t handle);
  ~HipDNNGraph();

  // Non-copyable
  HipDNNGraph(const HipDNNGraph&) = delete;
  HipDNNGraph& operator=(const HipDNNGraph&) = delete;

  /// @brief Build the graph from ONNX graph inputs, outputs, and nodes
  ///
  /// This builds the hipDNN graph structure but does not compile it.
  /// Call Compile() after Build() to finalize the graph for execution.
  ///
  /// @param graph_inputs The graph input value infos
  /// @param graph_outputs The graph output value infos
  /// @param nodes The nodes to process
  Status Build(const std::vector<Ort::ConstValueInfo>& graph_inputs,
               const std::vector<Ort::ConstValueInfo>& graph_outputs,
               const std::vector<Ort::ConstNode>& nodes);

#ifdef HIPDNN_EP_HAS_TORCH_MLIR
  /// @brief Build the graph from an MLIR hipdnn.graph region
  ///
  /// This builds the hipDNN graph structure but does not compile it.
  /// Call Compile() after Build() to finalize the graph for execution.
  ///
  /// @param region The region containing torch.aten ops
  /// @return Status::Success() on success, Status::Failure() with message on error
  Status Build(mlir::Region& region);
#endif

  /// @brief Compile the built graph for execution
  ///
  /// Must be called after Build(). Validates the graph, creates execution plans,
  /// and allocates workspace.
  ///
  /// @return Status::Success() on success, Status::Failure() with message on error
  Status Compile();

  /// @brief Execute the graph
  /// @param kernel_ctx The ORT kernel context with input/output tensors
  Status Execute(OrtKernelContext* kernel_ctx);

 private:
  std::unique_ptr<HipDNNGraphImpl> impl_;
};

}  // namespace hipdnn_ep
