// Copyright (c) 2025, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "hipdnn_ep/blas_graph/blas_graph.h"
#include "hipdnn_ep/core/ep_utils.h"
#include "hipdnn_ep/hipdnn_graph/hipdnn_graph.h"
#include "hipdnn_ep/torch_mlir_graph/ir_builder.h"

#include <memory>
#include <vector>

namespace hipdnn_ep {

/// @brief Configuration for Kernel construction using builder pattern
struct KernelConfig {
  // hipDNN
  hipdnnHandle_t getHipDNNHandle() const { return hipdnn_handle_; }
  KernelConfig& setHipDNNHandle(hipdnnHandle_t handle) {
    hipdnn_handle_ = handle;
    return *this;
  }
  bool useHipDNN() const { return hipdnn_handle_ != nullptr; }

  // hipBLAS-LT
  hipblaslt_handle_t getHipBlasLtHandle() const { return hipblaslt_handle_; }
  KernelConfig& setHipBlasLtHandle(hipblaslt_handle_t handle) {
    hipblaslt_handle_ = handle;
    return *this;
  }
  bool useHipBlasLT() const { return hipblaslt_handle_ != nullptr; }

  // Torch-MLIR
  bool dumpTorchMlir() const { return dump_torch_mlir_; }
  KernelConfig& setUseTorchMlir(bool dump = false) {
    use_torch_mlir_ = true;
    dump_torch_mlir_ = dump;
    return *this;
  }
  bool useTorchMlir() const { return use_torch_mlir_; }

 private:
  hipdnnHandle_t hipdnn_handle_{nullptr};
  hipblaslt_handle_t hipblaslt_handle_{nullptr};
  bool dump_torch_mlir_{false};
  bool use_torch_mlir_{false};
};

/// @brief Generic kernel that builds and executes hipDNN graphs or hipBLAS-LT matmuls
struct Kernel {
  Kernel(const OrtApi& ort_api, const OrtLogger& logger, KernelConfig&& config);
  ~Kernel();

  /// @brief Build and compile graph from an ORT graph
  OrtStatus* BuildAndCompile(Ort::ConstGraph graph);

  /// @brief Execute the compiled graph
  OrtStatus* Execute(OrtKernelContext* kernel_ctx);

 private:
  const OrtApi& ort_api_;
  const OrtLogger& logger_;
  KernelConfig config_;

  // hipDNN graph (nullptr when using blas_graph_)
  std::unique_ptr<HipDNNGraph> hipdnn_graph_;

  // hipBLAS-LT support (nullptr if unavailable or not used)
  std::unique_ptr<BlasGraph> blas_graph_;

  // Torch-MLIR support
  std::unique_ptr<IRBuilder> ir_builder_;
};

}  // namespace hipdnn_ep
