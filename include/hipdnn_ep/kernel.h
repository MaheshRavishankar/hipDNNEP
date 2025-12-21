// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// hipDNN includes
#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>

namespace hipdnn_ep {

/// @brief Generic kernel that builds and executes hipDNN graphs
struct Kernel {
  Kernel(const OrtApi& ort_api, const OrtLogger& logger, hipdnnHandle_t handle);
  ~Kernel();

  /// @brief Build and compile hipDNN graph from an ORT graph
  OrtStatus* BuildAndCompile(Ort::ConstGraph graph);

  /// @brief Execute the compiled graph
  OrtStatus* Execute(OrtKernelContext* kernel_ctx);

 private:
  /// @brief Compile the hipDNN graph after all ops are added
  OrtStatus* CompileGraph();

  const OrtApi& ort_api_;
  const OrtLogger& logger_;
  hipdnnHandle_t handle_;

  // hipDNN graph
  std::unique_ptr<hipdnn_frontend::graph::Graph> graph_;

  // Workspace
  std::vector<char> workspace_;

  // Graph input/output info (stored at compile time)
  std::vector<int64_t> input_uids_;   // UID for each graph input
  std::vector<int64_t> output_uids_;  // UID for each graph output
  std::vector<std::vector<int64_t>> output_shapes_;

  // Symbol table: maps value name to TensorAttributes
  using TensorAttrPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;
  std::unordered_map<std::string, TensorAttrPtr> symbol_table_;

  // UID counter for tensor attributes
  int64_t next_uid_{1};
};

}  // namespace hipdnn_ep
