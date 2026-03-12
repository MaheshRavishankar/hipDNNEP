// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "hipdnn_ep/core/ort_api.h"
#include "hipdnn_ep/hipdnn_graph/ep_status.h"

#include <memory>
#include <vector>

namespace hipdnn_ep {

// Forward declaration for pimpl
struct ReductionGraphImpl;

/// @brief Reduction modes supported by the reduction graph.
///
/// These map to ONNX Reduce* ops and to Fusilli's ReductionAttr::Mode.
/// New modes can be added here as the backend gains support.
enum class ReductionMode {
  kSum,
  kMax,
  kMin,
};

/// @brief Encapsulates a GPU reduction operation for the hipDNN EP.
///
/// This class provides Build/Compile/Execute methods matching the HipDNNGraph
/// pattern.  Because the hipDNN frontend graph API does not yet expose a
/// reduction node, this implementation uses a simple HIP kernel to perform
/// the reduction directly.
class ReductionGraph {
 public:
  ReductionGraph();
  ~ReductionGraph();

  // Non-copyable
  ReductionGraph(const ReductionGraph&) = delete;
  ReductionGraph& operator=(const ReductionGraph&) = delete;

  /// @brief Build the reduction graph from ONNX graph inputs, outputs, and nodes.
  ///
  /// The graph must contain exactly one ReduceSum/ReduceMax/ReduceMin node.
  Status Build(const std::vector<Ort::ConstValueInfo>& graph_inputs,
               const std::vector<Ort::ConstValueInfo>& graph_outputs,
               const std::vector<Ort::ConstNode>& nodes);

  /// @brief Compile (no-op for the HIP kernel path; kept for API symmetry).
  Status Compile();

  /// @brief Execute the reduction.
  Status Execute(OrtKernelContext* kernel_ctx);

 private:
  std::unique_ptr<ReductionGraphImpl> impl_;
};

}  // namespace hipdnn_ep
