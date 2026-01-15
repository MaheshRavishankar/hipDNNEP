// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// MIOpen includes
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>

namespace hipdnn_ep {

/// @brief Generic kernel that builds and executes operations using MIOpen
struct Kernel {
  Kernel(const OrtApi& ort_api, const OrtLogger& logger);
  ~Kernel();

  /// @brief Build and compile from an ORT graph
  OrtStatus* BuildAndCompile(Ort::ConstGraph graph);

  /// @brief Execute the compiled operations
  OrtStatus* Execute(OrtKernelContext* kernel_ctx);

 private:
  const OrtApi& ort_api_;
  const OrtLogger& logger_;

  // MIOpen handle
  miopenHandle_t miopen_handle_{nullptr};

  // Convolution descriptors
  miopenTensorDescriptor_t x_desc_{nullptr};  // Input
  miopenTensorDescriptor_t w_desc_{nullptr};  // Weights
  miopenTensorDescriptor_t y_desc_{nullptr};  // Output
  miopenTensorDescriptor_t b_desc_{nullptr};  // Bias (optional)
  miopenConvolutionDescriptor_t conv_desc_{nullptr};

  // Convolution algorithm and workspace
  miopenConvFwdAlgorithm_t conv_algo_{miopenConvolutionFwdAlgoGEMM};
  size_t workspace_size_{0};
  void* workspace_{nullptr};

  // Tensor shapes
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> w_shape_;
  std::vector<int64_t> y_shape_;
  std::vector<int64_t> b_shape_;

  // Graph I/O info
  size_t num_inputs_{0};
  size_t num_outputs_{0};
  std::vector<std::vector<int64_t>> output_shapes_;

  // Bias support
  bool has_bias_{false};

  // Data type
  miopenDataType_t data_type_{miopenFloat};
};

}  // namespace hipdnn_ep
