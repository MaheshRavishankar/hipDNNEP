// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "hipdnn_ep/hipdnn_graph/ep_status.h"

#include "hipdnn_ep/core/ort_api.h"

#include <memory>
#include <vector>

namespace hipdnn_ep {

// Forward declaration for pimpl
struct PointwiseGraphImpl;

/// @brief Encapsulates binary pointwise operations (Mul, Sub, Add, Div)
///
/// Uses HIP kernels directly since hipDNN's graph API requires a compute
/// operation (conv, matmul) as the primary node and does not support
/// standalone pointwise graphs.
class PointwiseGraph {
 public:
  PointwiseGraph();
  ~PointwiseGraph();

  // Non-copyable
  PointwiseGraph(const PointwiseGraph&) = delete;
  PointwiseGraph& operator=(const PointwiseGraph&) = delete;

  /// @brief Build from ONNX graph inputs, outputs, and nodes
  /// @return Status::Success() on success, Status::Failure() on error
  Status Build(const std::vector<Ort::ConstValueInfo>& graph_inputs,
               const std::vector<Ort::ConstValueInfo>& graph_outputs,
               const std::vector<Ort::ConstNode>& nodes);

  /// @brief Compile the graph (no-op for pointwise, kept for API consistency)
  Status Compile();

  /// @brief Execute the pointwise operation
  Status Execute(OrtKernelContext* kernel_ctx);

 private:
  std::unique_ptr<PointwiseGraphImpl> impl_;
};

}  // namespace hipdnn_ep
