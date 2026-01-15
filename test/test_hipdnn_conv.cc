// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

// Standalone hipDNN test to demonstrate direct hipDNN frontend API usage
// for conv and conv+bias operations

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <memory>

// Helper macro to check HIP status
#define HIP_CHECK(call)                                                 \
  do {                                                                   \
    hipError_t err = (call);                                            \
    ASSERT_EQ(err, hipSuccess)                                          \
        << "HIP error: " << hipGetErrorString(err);                     \
  } while (0)

class HipDNNConvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create hipDNN handle
    hipdnnStatus_t status = hipdnnCreate(&handle_);
    if (status != HIPDNN_STATUS_SUCCESS) {
      handle_ = nullptr;
      GTEST_SKIP() << "Failed to create hipDNN handle";
    }
  }

  void TearDown() override {
    if (handle_) {
      hipdnnDestroy(handle_);
      handle_ = nullptr;
    }
  }

  hipdnnHandle_t handle_{nullptr};
};

// Reference CPU implementation for verification
void ReferenceConv2D(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w) {
  int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

  for (int n = 0; n < N; ++n) {
    for (int c_out = 0; c_out < C_out; ++c_out) {
      for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
          float sum = 0.0f;

          for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int k_h = 0; k_h < K_h; ++k_h) {
              for (int k_w = 0; k_w < K_w; ++k_w) {
                int h_in = h_out * stride_h - pad_h + k_h;
                int w_in = w_out * stride_w - pad_w + k_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                  int input_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                  int weight_idx = c_out * C_in * K_h * K_w + c_in * K_h * K_w + k_h * K_w + k_w;
                  sum += input[input_idx] * weight[weight_idx];
                }
              }
            }
          }

          // Add bias if present
          if (bias != nullptr) {
            sum += bias[c_out];
          }

          int output_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
          output[output_idx] = sum;
        }
      }
    }
  }
}

// Helper to compute NCHW strides
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return strides;
}

// Test: hipDNN Convolution without bias
TEST_F(HipDNNConvTest, ConvolutionWithoutBias) {
  ASSERT_NE(handle_, nullptr) << "hipDNN handle not available";

  // Test parameters
  const int N = 1, C_in = 3, H_in = 8, W_in = 8;
  const int C_out = 2, K_h = 3, K_w = 3;
  const int pad_h = 1, pad_w = 1;
  const int stride_h = 1, stride_w = 1;
  const int dilation_h = 1, dilation_w = 1;

  const int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
  const int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

  // Define tensor dimensions
  std::vector<int64_t> x_dims = {N, C_in, H_in, W_in};
  std::vector<int64_t> w_dims = {C_out, C_in, K_h, K_w};
  std::vector<int64_t> y_dims = {N, C_out, H_out, W_out};

  auto x_strides = ComputeStrides(x_dims);
  auto w_strides = ComputeStrides(w_dims);
  auto y_strides = ComputeStrides(y_dims);

  // Allocate host memory
  size_t x_size = N * C_in * H_in * W_in;
  size_t w_size = C_out * C_in * K_h * K_w;
  size_t y_size = N * C_out * H_out * W_out;

  std::vector<float> h_x(x_size);
  std::vector<float> h_w(w_size);
  std::vector<float> h_y(y_size, 0.0f);
  std::vector<float> h_y_ref(y_size, 0.0f);

  // Initialize with random data
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h_x) v = dist(gen);
  for (auto& v : h_w) v = dist(gen);

  // Compute reference result
  ReferenceConv2D(h_x.data(), h_w.data(), nullptr, h_y_ref.data(),
                  N, C_in, H_in, W_in, C_out, K_h, K_w,
                  pad_h, pad_w, stride_h, stride_w);

  // Allocate device memory
  float *d_x, *d_w, *d_y;
  HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, w_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, y_size * sizeof(float)));

  // Copy input data to device
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), w_size * sizeof(float), hipMemcpyHostToDevice));

  // Build hipDNN graph
  std::cout << "Building hipDNN graph for convolution..." << std::endl;

  // Define unique IDs for tensors
  constexpr int64_t X_UID = 1;
  constexpr int64_t W_UID = 2;
  constexpr int64_t Y_UID = 3;

  hipdnn_frontend::graph::Graph graph;

  // Create input tensor X
  auto x_tensor = graph.tensor(
      hipdnn_frontend::graph::TensorAttributes()
          .set_uid(X_UID)
          .set_name("X")
          .set_data_type(hipdnn_frontend::DataType::FLOAT)
          .set_dim(x_dims)
          .set_stride(x_strides));

  // Create weight tensor W
  auto w_tensor = graph.tensor(
      hipdnn_frontend::graph::TensorAttributes()
          .set_uid(W_UID)
          .set_name("W")
          .set_data_type(hipdnn_frontend::DataType::FLOAT)
          .set_dim(w_dims)
          .set_stride(w_strides));

  // Create convolution operation
  auto y_tensor = graph.conv_fprop(
      x_tensor, w_tensor,
      hipdnn_frontend::graph::ConvFpropAttributes()
          .set_padding({pad_h, pad_w})
          .set_stride({stride_h, stride_w})
          .set_dilation({dilation_h, dilation_w})
          .set_compute_data_type(hipdnn_frontend::DataType::FLOAT));

  // Set output tensor attributes
  y_tensor->set_uid(Y_UID)
      .set_name("Y")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(y_dims)
      .set_stride(y_strides)
      .set_output(true);

  // Validate the graph
  std::cout << "Validating graph..." << std::endl;
  auto status = graph.validate();
  if (!status.is_good()) {
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_y);
    FAIL() << "hipDNN graph validation failed: " << status.get_message();
  }

  // Build operation graph
  std::cout << "Building operation graph..." << std::endl;
  status = graph.build_operation_graph(handle_);
  if (!status.is_good()) {
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_y);
    FAIL() << "hipDNN build_operation_graph failed: " << status.get_message();
  }

  // Create execution plans
  std::cout << "Creating execution plans..." << std::endl;
  status = graph.create_execution_plans({hipdnn_frontend::HeuristicMode::FALLBACK});
  if (!status.is_good()) {
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_y);
    FAIL() << "hipDNN create_execution_plans failed: " << status.get_message();
  }

  // Build plans
  std::cout << "Building plans..." << std::endl;
  status = graph.build_plans();
  if (!status.is_good()) {
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_y);
    FAIL() << "hipDNN build_plans failed: " << status.get_message();
  }

  // Allocate workspace (use a reasonable default size)
  void* workspace = nullptr;
  size_t workspace_size = 64 * 1024 * 1024;  // 64 MB default
  HIP_CHECK(hipMalloc(&workspace, workspace_size));
  std::cout << "Allocated workspace: " << workspace_size << " bytes" << std::endl;

  // Create variant pack with tensor pointers
  std::unordered_map<int64_t, void*> variant_pack = {
      {X_UID, d_x},
      {W_UID, d_w},
      {Y_UID, d_y}
  };

  // Execute
  std::cout << "Executing hipDNN convolution..." << std::endl;
  status = graph.execute(handle_, variant_pack, workspace);
  if (!status.is_good()) {
    hipFree(workspace);
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_y);
    FAIL() << "hipDNN execute failed: " << status.get_message();
  }

  HIP_CHECK(hipDeviceSynchronize());

  // Copy result back to host
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, y_size * sizeof(float), hipMemcpyDeviceToHost));

  // Compare with reference
  float max_diff = 0.0f;
  for (size_t i = 0; i < h_y.size(); ++i) {
    float diff = std::abs(h_y[i] - h_y_ref[i]);
    max_diff = std::max(max_diff, diff);
    EXPECT_NEAR(h_y[i], h_y_ref[i], 1e-4f)
        << "Mismatch at index " << i << ": GPU=" << h_y[i] << ", CPU=" << h_y_ref[i];
  }
  std::cout << "Max difference (conv without bias): " << max_diff << std::endl;

  // Cleanup
  hipFree(workspace);
  hipFree(d_x);
  hipFree(d_w);
  hipFree(d_y);
}

// Test: hipDNN Convolution with bias (using pointwise ADD)
TEST_F(HipDNNConvTest, ConvolutionWithBias) {
  ASSERT_NE(handle_, nullptr) << "hipDNN handle not available";

  // Test parameters
  const int N = 1, C_in = 3, H_in = 8, W_in = 8;
  const int C_out = 2, K_h = 3, K_w = 3;
  const int pad_h = 1, pad_w = 1;
  const int stride_h = 1, stride_w = 1;
  const int dilation_h = 1, dilation_w = 1;

  const int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
  const int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

  // Define tensor dimensions
  std::vector<int64_t> x_dims = {N, C_in, H_in, W_in};
  std::vector<int64_t> w_dims = {C_out, C_in, K_h, K_w};
  std::vector<int64_t> y_dims = {N, C_out, H_out, W_out};
  std::vector<int64_t> b_dims = {1, C_out, 1, 1};  // Bias shape for broadcasting

  auto x_strides = ComputeStrides(x_dims);
  auto w_strides = ComputeStrides(w_dims);
  auto y_strides = ComputeStrides(y_dims);
  // Broadcast strides for bias: only channel dimension has non-zero stride
  std::vector<int64_t> b_strides = {0, 1, 0, 0};

  // Allocate host memory
  size_t x_size = N * C_in * H_in * W_in;
  size_t w_size = C_out * C_in * K_h * K_w;
  size_t y_size = N * C_out * H_out * W_out;
  size_t b_size = C_out;

  std::vector<float> h_x(x_size);
  std::vector<float> h_w(w_size);
  std::vector<float> h_b(b_size);
  std::vector<float> h_y(y_size, 0.0f);
  std::vector<float> h_y_ref(y_size, 0.0f);

  // Initialize with random data
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h_x) v = dist(gen);
  for (auto& v : h_w) v = dist(gen);
  for (auto& v : h_b) v = dist(gen);

  // Compute reference result (conv + bias)
  ReferenceConv2D(h_x.data(), h_w.data(), h_b.data(), h_y_ref.data(),
                  N, C_in, H_in, W_in, C_out, K_h, K_w,
                  pad_h, pad_w, stride_h, stride_w);

  // Allocate device memory
  float *d_x, *d_w, *d_b, *d_y;
  HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, w_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, b_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, y_size * sizeof(float)));

  // Copy input data to device
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), w_size * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), b_size * sizeof(float), hipMemcpyHostToDevice));

  // Build hipDNN graph
  std::cout << "Building hipDNN graph for conv + bias..." << std::endl;

  // Define unique IDs for tensors
  constexpr int64_t X_UID = 1;
  constexpr int64_t W_UID = 2;
  constexpr int64_t B_UID = 3;
  constexpr int64_t CONV_OUT_UID = 4;  // Intermediate conv output
  constexpr int64_t Y_UID = 5;         // Final output after bias

  hipdnn_frontend::graph::Graph graph;

  // Create input tensor X
  auto x_tensor = graph.tensor(
      hipdnn_frontend::graph::TensorAttributes()
          .set_uid(X_UID)
          .set_name("X")
          .set_data_type(hipdnn_frontend::DataType::FLOAT)
          .set_dim(x_dims)
          .set_stride(x_strides));

  // Create weight tensor W
  auto w_tensor = graph.tensor(
      hipdnn_frontend::graph::TensorAttributes()
          .set_uid(W_UID)
          .set_name("W")
          .set_data_type(hipdnn_frontend::DataType::FLOAT)
          .set_dim(w_dims)
          .set_stride(w_strides));

  // Create bias tensor B with broadcasting strides
  auto b_tensor = graph.tensor(
      hipdnn_frontend::graph::TensorAttributes()
          .set_uid(B_UID)
          .set_name("B")
          .set_data_type(hipdnn_frontend::DataType::FLOAT)
          .set_dim(b_dims)
          .set_stride(b_strides));

  // Step 1: Create convolution operation -> intermediate output
  auto conv_out_tensor = graph.conv_fprop(
      x_tensor, w_tensor,
      hipdnn_frontend::graph::ConvFpropAttributes()
          .set_padding({pad_h, pad_w})
          .set_stride({stride_h, stride_w})
          .set_dilation({dilation_h, dilation_w})
          .set_compute_data_type(hipdnn_frontend::DataType::FLOAT));

  // Set intermediate tensor attributes
  conv_out_tensor->set_uid(CONV_OUT_UID)
      .set_name("ConvOut")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(y_dims)
      .set_stride(y_strides)
      .set_is_virtual(true);  // Virtual tensor - no explicit allocation needed

  // Step 2: Add bias using pointwise ADD -> final output
  auto y_tensor = graph.pointwise(
      conv_out_tensor, b_tensor,
      hipdnn_frontend::graph::PointwiseAttributes()
          .set_mode(hipdnn_frontend::PointwiseMode::ADD)
          .set_compute_data_type(hipdnn_frontend::DataType::FLOAT));

  // Set final output tensor attributes
  y_tensor->set_uid(Y_UID)
      .set_name("Y")
      .set_data_type(hipdnn_frontend::DataType::FLOAT)
      .set_dim(y_dims)
      .set_stride(y_strides)
      .set_output(true);

  // Validate the graph
  std::cout << "Validating graph..." << std::endl;
  auto status = graph.validate();
  if (!status.is_good()) {
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_b);
    hipFree(d_y);
    FAIL() << "hipDNN graph validation failed: " << status.get_message();
  }

  // Build operation graph
  std::cout << "Building operation graph..." << std::endl;
  status = graph.build_operation_graph(handle_);
  if (!status.is_good()) {
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_b);
    hipFree(d_y);
    FAIL() << "hipDNN build_operation_graph failed: " << status.get_message();
  }

  // Create execution plans
  std::cout << "Creating execution plans..." << std::endl;
  status = graph.create_execution_plans({hipdnn_frontend::HeuristicMode::FALLBACK});
  if (!status.is_good()) {
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_b);
    hipFree(d_y);
    FAIL() << "hipDNN create_execution_plans failed: " << status.get_message();
  }

  // Build plans
  std::cout << "Building plans..." << std::endl;
  status = graph.build_plans();
  if (!status.is_good()) {
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_b);
    hipFree(d_y);
    FAIL() << "hipDNN build_plans failed: " << status.get_message();
  }

  // Allocate workspace (use a reasonable default size)
  void* workspace = nullptr;
  size_t workspace_size = 64 * 1024 * 1024;  // 64 MB default
  HIP_CHECK(hipMalloc(&workspace, workspace_size));
  std::cout << "Allocated workspace: " << workspace_size << " bytes" << std::endl;

  // Create variant pack with tensor pointers
  // Note: Virtual tensors (CONV_OUT_UID) don't need to be in the variant pack
  std::unordered_map<int64_t, void*> variant_pack = {
      {X_UID, d_x},
      {W_UID, d_w},
      {B_UID, d_b},
      {Y_UID, d_y}
  };

  // Execute
  std::cout << "Executing hipDNN conv + bias..." << std::endl;
  status = graph.execute(handle_, variant_pack, workspace);
  if (!status.is_good()) {
    hipFree(workspace);
    hipFree(d_x);
    hipFree(d_w);
    hipFree(d_b);
    hipFree(d_y);
    FAIL() << "hipDNN execute failed: " << status.get_message();
  }

  HIP_CHECK(hipDeviceSynchronize());

  // Copy result back to host
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, y_size * sizeof(float), hipMemcpyDeviceToHost));

  // Compare with reference
  float max_diff = 0.0f;
  for (size_t i = 0; i < h_y.size(); ++i) {
    float diff = std::abs(h_y[i] - h_y_ref[i]);
    max_diff = std::max(max_diff, diff);
    EXPECT_NEAR(h_y[i], h_y_ref[i], 1e-4f)
        << "Mismatch at index " << i << ": GPU=" << h_y[i] << ", CPU=" << h_y_ref[i];
  }
  std::cout << "Max difference (conv with bias): " << max_diff << std::endl;

  // Cleanup
  hipFree(workspace);
  hipFree(d_x);
  hipFree(d_w);
  hipFree(d_b);
  hipFree(d_y);
}
