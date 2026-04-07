// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "test/common/ep_test_base.h"

#ifndef MATMUL_TEST_MODEL_PATH
#define MATMUL_TEST_MODEL_PATH "./matmul_test.onnx"
#endif

#ifndef GEMM_TEST_MODEL_PATH
#define GEMM_TEST_MODEL_PATH "./gemm_test.onnx"
#endif

#ifndef GEMM_BIAS_TEST_MODEL_PATH
#define GEMM_BIAS_TEST_MODEL_PATH "./gemm_bias_test.onnx"
#endif

#ifndef GEMM_TRANS_A_TEST_MODEL_PATH
#define GEMM_TRANS_A_TEST_MODEL_PATH "./gemm_trans_a_test.onnx"
#endif

#ifndef GEMM_TRANS_B_TEST_MODEL_PATH
#define GEMM_TRANS_B_TEST_MODEL_PATH "./gemm_trans_b_test.onnx"
#endif

#ifndef GEMM_SCALED_TEST_MODEL_PATH
#define GEMM_SCALED_TEST_MODEL_PATH "./gemm_scaled_test.onnx"
#endif

#ifndef GEMM_SCALAR_BIAS_TEST_MODEL_PATH
#define GEMM_SCALAR_BIAS_TEST_MODEL_PATH "./gemm_scalar_bias_test.onnx"
#endif

class HipDNNMatMulTest : public HipDNNTestBase {};

// Reference MatMul implementation for testing
static void ReferenceMatMul(const float* a, const float* b, float* y,
                            int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      y[i * n + j] = sum;
    }
  }
}

// Reference Gemm implementation for testing
static void ReferenceGemm(const float* a, const float* b, const float* c,
                          float* y, int m, int k, int n, bool trans_a,
                          bool trans_b, float alpha, float beta) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        float a_val = trans_a ? a[l * m + i] : a[i * k + l];
        float b_val = trans_b ? b[j * k + l] : b[l * n + j];
        sum += a_val * b_val;
      }
      float c_val = (c != nullptr) ? c[i * n + j] : 0.0f;
      y[i * n + j] = alpha * sum + beta * c_val;
    }
  }
}

TEST_F(HipDNNMatMulTest, ReferenceMatMulCorrectness) {
  const int m = 2, k = 3, n = 2;
  std::vector<float> a = {1, 2, 3, 4, 5, 6};     // 2x3
  std::vector<float> b = {7, 8, 9, 10, 11, 12};  // 3x2
  std::vector<float> y(m * n, 0.0f);

  ReferenceMatMul(a.data(), b.data(), y.data(), m, k, n);

  EXPECT_NEAR(y[0], 58.0f, 1e-5f);
  EXPECT_NEAR(y[1], 64.0f, 1e-5f);
  EXPECT_NEAR(y[2], 139.0f, 1e-5f);
  EXPECT_NEAR(y[3], 154.0f, 1e-5f);
}

TEST_F(HipDNNMatMulTest, BasicMatMul) {
  const int64_t m = 64, k = 128, n = 32;

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  RunAndCompare(MATMUL_TEST_MODEL_PATH, {a_data, b_data},
                {{m, k}, {k, n}}, {"A", "B"}, "Y");
}

TEST_F(HipDNNMatMulTest, BasicGemm) {
  const int64_t m = 64, k = 128, n = 32;

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  RunAndCompare(GEMM_TEST_MODEL_PATH, {a_data, b_data},
                {{m, k}, {k, n}}, {"A", "B"}, "Y");
}

TEST_F(HipDNNMatMulTest, GemmWithBias) {
  const int64_t m = 64, k = 128, n = 32;

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);
  std::vector<float> c_data(m * n);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }
  for (size_t i = 0; i < c_data.size(); ++i) {
    c_data[i] = static_cast<float>((i + 5) % 10) / 10.0f;
  }

  RunAndCompare(GEMM_BIAS_TEST_MODEL_PATH, {a_data, b_data, c_data},
                {{m, k}, {k, n}, {m, n}}, {"A", "B", "C"}, "Y");
}

TEST_F(HipDNNMatMulTest, GemmWithTransposeA) {
  const int64_t m = 64, k = 128, n = 32;
  // With transA=1: A shape is [k, m] = [128, 64]
  std::vector<float> a_data(k * m);
  std::vector<float> b_data(k * n);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  RunAndCompare(GEMM_TRANS_A_TEST_MODEL_PATH, {a_data, b_data},
                {{k, m}, {k, n}}, {"A", "B"}, "Y");
}

TEST_F(HipDNNMatMulTest, GemmWithTransposeB) {
  const int64_t m = 64, k = 128, n = 32;
  // With transB=1: B shape is [n, k] = [32, 128]
  std::vector<float> a_data(m * k);
  std::vector<float> b_data(n * k);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  RunAndCompare(GEMM_TRANS_B_TEST_MODEL_PATH, {a_data, b_data},
                {{m, k}, {n, k}}, {"A", "B"}, "Y");
}

TEST_F(HipDNNMatMulTest, GemmWithScaling) {
  const int64_t m = 64, k = 128, n = 32;

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);
  std::vector<float> c_data(m * n);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }
  for (size_t i = 0; i < c_data.size(); ++i) {
    c_data[i] = static_cast<float>((i + 5) % 10) / 10.0f;
  }

  RunAndCompare(GEMM_SCALED_TEST_MODEL_PATH, {a_data, b_data, c_data},
                {{m, k}, {k, n}, {m, n}}, {"A", "B", "C"}, "Y");
}

TEST_F(HipDNNMatMulTest, GemmWithScalarBias) {
  // Model: Y = A @ B + 0.5  (scalar bias is a constant initializer)
  // Only A and B are runtime inputs.
  const int64_t m = 64, k = 128, n = 32;

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  RunAndCompare(GEMM_SCALAR_BIAS_TEST_MODEL_PATH, {a_data, b_data},
                {{m, k}, {k, n}}, {"A", "B"}, "Y");
}
