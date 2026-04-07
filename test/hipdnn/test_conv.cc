// Copyright (c) 2025, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "test/common/ep_test_base.h"

#ifndef CONV_TEST_MODEL_PATH
#define CONV_TEST_MODEL_PATH "./conv_test.onnx"
#endif

#ifndef CONV_BIAS_TEST_MODEL_PATH
#define CONV_BIAS_TEST_MODEL_PATH "./conv_test_bias.onnx"
#endif

class HipDNNConvTest : public HipDNNTestBase {};

// Simple reference Conv2D implementation for verification
static void ReferenceConv2D(
    const float* input, const float* weight, float* output,
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

          int output_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
          output[output_idx] = sum;
        }
      }
    }
  }
}

TEST_F(HipDNNConvTest, BasicConv2D) {
  const int64_t N = 1, C = 1, H = 8, W = 8;
  const std::vector<int64_t> input_shape = {N, C, H, W};
  const size_t input_size = N * C * H * W;

  std::vector<float> input_data(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input_data[i] = static_cast<float>(i % 10) / 10.0f;
  }

  RunAndCompare(CONV_TEST_MODEL_PATH, {input_data}, {input_shape}, {"X"},
                "Y");
}

TEST_F(HipDNNConvTest, ConvWithBias) {
  const int64_t N = 1, C = 1, H = 8, W = 8;
  const std::vector<int64_t> input_shape = {N, C, H, W};
  const size_t input_size = N * C * H * W;

  std::vector<float> input_data(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input_data[i] = static_cast<float>(i % 10) / 10.0f;
  }

  RunAndCompare(CONV_BIAS_TEST_MODEL_PATH, {input_data}, {input_shape}, {"X"},
                "Y");
}

TEST_F(HipDNNConvTest, ReferenceConvCorrectness) {
  // Test the reference implementation
  const int N = 1, C_in = 1, H_in = 4, W_in = 4;
  const int C_out = 1, K_h = 3, K_w = 3;
  const int pad_h = 0, pad_w = 0;
  const int stride_h = 1, stride_w = 1;

  // Simple input: 4x4 matrix of ones
  std::vector<float> input(N * C_in * H_in * W_in, 1.0f);

  // Simple weight: 3x3 matrix of ones
  std::vector<float> weight(C_out * C_in * K_h * K_w, 1.0f);

  // Output should be 2x2 (4 - 3 + 1 = 2)
  int H_out = (H_in - K_h) / stride_h + 1;
  int W_out = (W_in - K_w) / stride_w + 1;
  std::vector<float> output(N * C_out * H_out * W_out, 0.0f);

  ReferenceConv2D(input.data(), weight.data(), output.data(),
                  N, C_in, H_in, W_in, C_out, K_h, K_w,
                  pad_h, pad_w, stride_h, stride_w);

  // Each output should be sum of 3x3 = 9 ones = 9.0
  for (int i = 0; i < static_cast<int>(output.size()); ++i) {
    EXPECT_NEAR(output[i], 9.0f, 1e-5f) << "Output mismatch at index " << i;
  }
}
