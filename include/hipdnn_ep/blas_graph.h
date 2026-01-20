// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"

#include <memory>
#include <vector>

// Opaque handle type for hipBLAS-LT (actual type hidden in blas_graph.cc)
using hipblaslt_handle_t = void*;

namespace hipdnn_ep {

// Forward declaration for pimpl
struct BlasGraphImpl;

/// @brief Encapsulates a hipBLAS-LT execution graph for MatMul/Gemm operations
///
/// This class mirrors hipDNN's Graph interface, providing Build and Execute
/// methods. All hipBLAS-LT specific types are hidden in the implementation.
class BlasGraph {
 public:
  /// @param handle hipBLAS-LT handle
  BlasGraph(const OrtApi& ort_api, hipblaslt_handle_t handle);
  ~BlasGraph();

  // Non-copyable
  BlasGraph(const BlasGraph&) = delete;
  BlasGraph& operator=(const BlasGraph&) = delete;

  /// @brief Build the graph from an ONNX MatMul or Gemm node
  /// @param node The ONNX node (must be MatMul or Gemm)
  /// @param output_shape The expected output shape [M, N]
  OrtStatus* Build(Ort::ConstNode node, const std::vector<int64_t>& output_shape);

  /// @brief Execute the graph
  /// @param kernel_ctx The ORT kernel context with input/output tensors
  OrtStatus* Execute(OrtKernelContext* kernel_ctx);

 private:
  std::unique_ptr<BlasGraphImpl> impl_;
};

/// @brief Create a hipBLAS-LT handle
/// @return The created handle, or nullptr on failure
hipblaslt_handle_t CreateHipBlasLtHandle();

/// @brief Destroy a hipBLAS-LT handle
/// @param handle The handle to destroy
void DestroyHipBlasLtHandle(hipblaslt_handle_t handle);

}  // namespace hipdnn_ep
