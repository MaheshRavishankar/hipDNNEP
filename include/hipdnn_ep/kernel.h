// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "blas_graph.h"
#include "ep_utils.h"
#include "hipdnn_graph.h"

#include <memory>
#include <vector>

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
  const OrtApi& ort_api_;
  const OrtLogger& logger_;

  // hipDNN graph (nullptr when using blas_graph_)
  hipdnnHandle_t handle_;
  std::unique_ptr<HipDNNGraph> hipdnn_graph_;

  // hipBLAS-LT support (nullptr if unavailable or not used)
  hipblaslt_handle_t hipblaslt_handle_{nullptr};
  std::unique_ptr<BlasGraph> blas_graph_;
};

}  // namespace hipdnn_ep
