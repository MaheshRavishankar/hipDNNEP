// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "blas_graph.h"
#include "ep_utils.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// hipDNN includes
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>

namespace hipdnn_ep {

/// @brief Generic kernel that builds and executes hipDNN graphs or hipBLAS-LT matmuls
struct Kernel {
  /// @param hipblaslt_handle Optional hipBLAS-LT handle for MatMul/Gemm support (nullptr if unavailable)
  Kernel(const OrtApi& ort_api, const OrtLogger& logger, hipdnnHandle_t handle,
         hipblaslt_handle_t hipblaslt_handle = nullptr);
  ~Kernel();

  /// @brief Build and compile graph from an ORT graph
  OrtStatus* BuildAndCompile(Ort::ConstGraph graph);

  /// @brief Execute the compiled graph
  OrtStatus* Execute(OrtKernelContext* kernel_ctx);

 private:
  /// @brief Compile the hipDNN graph after all ops are added
  OrtStatus* CompileGraph();

  const OrtApi& ort_api_;
  const OrtLogger& logger_;
  hipdnnHandle_t handle_;

  // hipDNN graph (nullptr when using blas_graph_)
  std::unique_ptr<hipdnn_frontend::graph::Graph> graph_;

  // Workspace for hipDNN graph
  std::vector<char> workspace_;

  // Graph input/output info (stored at compile time, used by hipDNN path)
  std::vector<int64_t> input_uids_;   // UID for each graph input
  std::vector<int64_t> output_uids_;  // UID for each graph output
  std::vector<std::vector<int64_t>> output_shapes_;

  // Symbol table: maps value name to TensorAttributes
  using TensorAttrPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;
  std::unordered_map<std::string, TensorAttrPtr> symbol_table_;

  // UID counter for tensor attributes
  int64_t next_uid_{1};

  // hipBLAS-LT support (nullptr if unavailable)
  hipblaslt_handle_t hipblaslt_handle_{nullptr};
  std::unique_ptr<BlasGraph> blas_graph_;
};

}  // namespace hipdnn_ep
