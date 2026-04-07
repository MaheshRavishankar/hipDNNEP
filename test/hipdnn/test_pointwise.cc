// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "test/common/ep_test_base.h"

#ifndef MUL_TEST_MODEL_PATH
#define MUL_TEST_MODEL_PATH "./mul_test.onnx"
#endif

#ifndef SUB_TEST_MODEL_PATH
#define SUB_TEST_MODEL_PATH "./sub_test.onnx"
#endif

#ifndef ADD_TEST_MODEL_PATH
#define ADD_TEST_MODEL_PATH "./add_test.onnx"
#endif

#ifndef DIV_TEST_MODEL_PATH
#define DIV_TEST_MODEL_PATH "./div_test.onnx"
#endif

#ifndef MUL_SCALAR_TEST_MODEL_PATH
#define MUL_SCALAR_TEST_MODEL_PATH "./mul_scalar_test.onnx"
#endif

#ifndef SUB_SCALAR_TEST_MODEL_PATH
#define SUB_SCALAR_TEST_MODEL_PATH "./sub_scalar_test.onnx"
#endif

#ifndef ADD_SCALAR_TEST_MODEL_PATH
#define ADD_SCALAR_TEST_MODEL_PATH "./add_scalar_test.onnx"
#endif

#ifndef DIV_SCALAR_TEST_MODEL_PATH
#define DIV_SCALAR_TEST_MODEL_PATH "./div_scalar_test.onnx"
#endif

class HipDNNPointwiseTest : public HipDNNTestBase {};

TEST_F(HipDNNPointwiseTest, Mul) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 0.1f, 0.01f);
  auto b = GenerateTestData(n, 1.0f, -0.003f);
  RunAndCompare(MUL_TEST_MODEL_PATH, {a, b}, {shape, shape}, {"A", "B"},
                "Y");
}

TEST_F(HipDNNPointwiseTest, Sub) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 5.0f, -0.01f);
  auto b = GenerateTestData(n, 0.5f, 0.02f);
  RunAndCompare(SUB_TEST_MODEL_PATH, {a, b}, {shape, shape}, {"A", "B"},
                "Y");
}

TEST_F(HipDNNPointwiseTest, Add) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, -1.0f, 0.01f);
  auto b = GenerateTestData(n, 2.0f, -0.005f);
  RunAndCompare(ADD_TEST_MODEL_PATH, {a, b}, {shape, shape}, {"A", "B"},
                "Y");
}

TEST_F(HipDNNPointwiseTest, Div) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 1.0f, 0.05f);
  auto b = GenerateTestData(n, 1.0f, 0.01f);
  RunAndCompare(DIV_TEST_MODEL_PATH, {a, b}, {shape, shape}, {"A", "B"},
                "Y", 1e-3f);
}

// Scalar-input tests: B is a scalar (shape [1]), A is a regular tensor.
TEST_F(HipDNNPointwiseTest, MulScalar) {
  const std::vector<int64_t> a_shape = {1, 4, 8, 8};
  const std::vector<int64_t> b_shape = {1};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 0.1f, 0.01f);
  std::vector<float> b = {2.5f};
  RunAndCompare(MUL_SCALAR_TEST_MODEL_PATH, {a, b}, {a_shape, b_shape},
                {"A", "B"}, "Y");
}

TEST_F(HipDNNPointwiseTest, SubScalar) {
  const std::vector<int64_t> a_shape = {1, 4, 8, 8};
  const std::vector<int64_t> b_shape = {1};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 5.0f, -0.01f);
  std::vector<float> b = {1.0f};
  RunAndCompare(SUB_SCALAR_TEST_MODEL_PATH, {a, b}, {a_shape, b_shape},
                {"A", "B"}, "Y");
}

TEST_F(HipDNNPointwiseTest, AddScalar) {
  const std::vector<int64_t> a_shape = {1, 4, 8, 8};
  const std::vector<int64_t> b_shape = {1};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, -1.0f, 0.01f);
  std::vector<float> b = {3.14f};
  RunAndCompare(ADD_SCALAR_TEST_MODEL_PATH, {a, b}, {a_shape, b_shape},
                {"A", "B"}, "Y");
}

TEST_F(HipDNNPointwiseTest, DivScalar) {
  const std::vector<int64_t> a_shape = {1, 4, 8, 8};
  const std::vector<int64_t> b_shape = {1};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 1.0f, 0.05f);
  std::vector<float> b = {2.0f};
  RunAndCompare(DIV_SCALAR_TEST_MODEL_PATH, {a, b}, {a_shape, b_shape},
                {"A", "B"}, "Y", 1e-3f);
}
