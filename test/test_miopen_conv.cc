// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

// Standalone MIOpen test to demonstrate direct conv and conv+bias API usage

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>

// Helper macro to check MIOpen status
#define MIOPEN_CHECK(call)                                              \
  do {                                                                   \
    miopenStatus_t status = (call);                                     \
    ASSERT_EQ(status, miopenStatusSuccess)                              \
        << "MIOpen error: " << status;                                  \
  } while (0)

// Helper macro to check HIP status
#define HIP_CHECK(call)                                                 \
  do {                                                                   \
    hipError_t err = (call);                                            \
    ASSERT_EQ(err, hipSuccess)                                          \
        << "HIP error: " << hipGetErrorString(err);                     \
  } while (0)

class MIOpenConvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create MIOpen handle
    miopenStatus_t status = miopenCreate(&handle_);
    if (status != miopenStatusSuccess) {
      handle_ = nullptr;
      GTEST_SKIP() << "Failed to create MIOpen handle";
    }
  }

  void TearDown() override {
    if (handle_) {
      miopenDestroy(handle_);
      handle_ = nullptr;
    }
  }

  miopenHandle_t handle_{nullptr};
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

// Test: MIOpen Convolution without bias
TEST_F(MIOpenConvTest, ConvolutionWithoutBias) {
  ASSERT_NE(handle_, nullptr) << "MIOpen handle not available";

  // Test parameters
  const int N = 1, C_in = 3, H_in = 8, W_in = 8;
  const int C_out = 2, K_h = 3, K_w = 3;
  const int pad_h = 1, pad_w = 1;
  const int stride_h = 1, stride_w = 1;
  const int dilation_h = 1, dilation_w = 1;

  const int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
  const int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

  // Allocate host memory
  std::vector<float> h_x(N * C_in * H_in * W_in);
  std::vector<float> h_w(C_out * C_in * K_h * K_w);
  std::vector<float> h_y(N * C_out * H_out * W_out, 0.0f);
  std::vector<float> h_y_ref(N * C_out * H_out * W_out, 0.0f);

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
  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, h_y.size() * sizeof(float)));

  // Copy input data to device
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), hipMemcpyHostToDevice));

  // Create tensor descriptors
  miopenTensorDescriptor_t x_desc, w_desc, y_desc;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&x_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&w_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&y_desc));

  MIOPEN_CHECK(miopenSet4dTensorDescriptor(x_desc, miopenFloat, N, C_in, H_in, W_in));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(w_desc, miopenFloat, C_out, C_in, K_h, K_w));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(y_desc, miopenFloat, N, C_out, H_out, W_out));

  // Create convolution descriptor
  miopenConvolutionDescriptor_t conv_desc;
  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                                pad_h, pad_w, stride_h, stride_w,
                                                dilation_h, dilation_w));

  // Get workspace size
  size_t workspace_size = 0;
  MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(handle_, w_desc, x_desc,
                                                         conv_desc, y_desc, &workspace_size));

  void* workspace = nullptr;
  if (workspace_size > 0) {
    HIP_CHECK(hipMalloc(&workspace, workspace_size));
  }

  // Find convolution algorithm (required by MIOpen)
  const int request_algo_count = 4;
  int returned_algo_count = 0;
  miopenConvAlgoPerf_t perf_results[request_algo_count];

  std::cout << "Finding convolution algorithm..." << std::endl;
  MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
      handle_, x_desc, d_x, w_desc, d_w, conv_desc, y_desc, d_y,
      request_algo_count, &returned_algo_count, perf_results,
      workspace, workspace_size, false));

  ASSERT_GT(returned_algo_count, 0) << "No algorithm found";
  miopenConvFwdAlgorithm_t algo = perf_results[0].fwd_algo;
  std::cout << "Selected algorithm: " << algo << std::endl;

  // Execute convolution: y = conv(x, w)
  float alpha = 1.0f, beta = 0.0f;
  std::cout << "Executing miopenConvolutionForward..." << std::endl;
  MIOPEN_CHECK(miopenConvolutionForward(handle_, &alpha, x_desc, d_x, w_desc, d_w,
                                         conv_desc, algo, &beta, y_desc, d_y,
                                         workspace, workspace_size));

  // Copy result back to host
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), hipMemcpyDeviceToHost));

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
  if (workspace) hipFree(workspace);
  hipFree(d_x);
  hipFree(d_w);
  hipFree(d_y);
  miopenDestroyTensorDescriptor(x_desc);
  miopenDestroyTensorDescriptor(w_desc);
  miopenDestroyTensorDescriptor(y_desc);
  miopenDestroyConvolutionDescriptor(conv_desc);
}

// Test: MIOpen Convolution with bias
TEST_F(MIOpenConvTest, ConvolutionWithBias) {
  ASSERT_NE(handle_, nullptr) << "MIOpen handle not available";

  // Test parameters
  const int N = 1, C_in = 3, H_in = 8, W_in = 8;
  const int C_out = 2, K_h = 3, K_w = 3;
  const int pad_h = 1, pad_w = 1;
  const int stride_h = 1, stride_w = 1;
  const int dilation_h = 1, dilation_w = 1;

  const int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
  const int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

  // Allocate host memory
  std::vector<float> h_x(N * C_in * H_in * W_in);
  std::vector<float> h_w(C_out * C_in * K_h * K_w);
  std::vector<float> h_b(C_out);
  std::vector<float> h_y(N * C_out * H_out * W_out, 0.0f);
  std::vector<float> h_y_ref(N * C_out * H_out * W_out, 0.0f);

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
  HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_w, h_w.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_b, h_b.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_y, h_y.size() * sizeof(float)));

  // Copy input data to device
  HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), hipMemcpyHostToDevice));

  // Create tensor descriptors
  miopenTensorDescriptor_t x_desc, w_desc, y_desc, b_desc;
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&x_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&w_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&y_desc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&b_desc));

  MIOPEN_CHECK(miopenSet4dTensorDescriptor(x_desc, miopenFloat, N, C_in, H_in, W_in));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(w_desc, miopenFloat, C_out, C_in, K_h, K_w));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(y_desc, miopenFloat, N, C_out, H_out, W_out));
  
  // Bias descriptor: [1, C_out, 1, 1] with strides for broadcasting
  int b_dims[4] = {1, C_out, 1, 1};
  int b_strides[4] = {C_out, 1, 1, 1};
  MIOPEN_CHECK(miopenSetTensorDescriptor(b_desc, miopenFloat, 4, b_dims, b_strides));

  // Create convolution descriptor
  miopenConvolutionDescriptor_t conv_desc;
  MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
  MIOPEN_CHECK(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                                pad_h, pad_w, stride_h, stride_w,
                                                dilation_h, dilation_w));

  // Get workspace size
  size_t workspace_size = 0;
  MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(handle_, w_desc, x_desc,
                                                         conv_desc, y_desc, &workspace_size));

  void* workspace = nullptr;
  if (workspace_size > 0) {
    HIP_CHECK(hipMalloc(&workspace, workspace_size));
  }

  // Find convolution algorithm (required by MIOpen)
  const int request_algo_count = 4;
  int returned_algo_count = 0;
  miopenConvAlgoPerf_t perf_results[request_algo_count];

  std::cout << "Finding convolution algorithm..." << std::endl;
  MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
      handle_, x_desc, d_x, w_desc, d_w, conv_desc, y_desc, d_y,
      request_algo_count, &returned_algo_count, perf_results,
      workspace, workspace_size, false));

  ASSERT_GT(returned_algo_count, 0) << "No algorithm found";
  miopenConvFwdAlgorithm_t algo = perf_results[0].fwd_algo;
  std::cout << "Selected algorithm: " << algo << std::endl;

  // Step 1: Execute convolution: y = conv(x, w)
  float alpha = 1.0f, beta = 0.0f;
  std::cout << "Step 1: Executing miopenConvolutionForward (y = conv(x, w))..." << std::endl;
  MIOPEN_CHECK(miopenConvolutionForward(handle_, &alpha, x_desc, d_x, w_desc, d_w,
                                         conv_desc, algo, &beta, y_desc, d_y,
                                         workspace, workspace_size));

  // Step 2: Add bias using miopenOpTensor: y = y + bias
  // Formula: C = alpha1*A + alpha2*B + beta*C
  // With alpha1=1, alpha2=1, beta=0: y = 1*y + 1*bias + 0*y = y + bias
  float alpha1 = 1.0f, alpha2 = 1.0f, beta_op = 0.0f;
  std::cout << "Step 2: Adding bias with miopenOpTensor (y = y + bias)..." << std::endl;
  MIOPEN_CHECK(miopenOpTensor(handle_, miopenTensorOpAdd,
                               &alpha1, y_desc, d_y,
                               &alpha2, b_desc, d_b,
                               &beta_op, y_desc, d_y));

  // Copy result back to host
  HIP_CHECK(hipMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), hipMemcpyDeviceToHost));

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
  if (workspace) hipFree(workspace);
  hipFree(d_x);
  hipFree(d_w);
  hipFree(d_b);
  hipFree(d_y);
  miopenDestroyTensorDescriptor(x_desc);
  miopenDestroyTensorDescriptor(w_desc);
  miopenDestroyTensorDescriptor(y_desc);
  miopenDestroyTensorDescriptor(b_desc);
  miopenDestroyConvolutionDescriptor(conv_desc);
}
