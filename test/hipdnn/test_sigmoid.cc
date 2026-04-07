// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "test/common/ep_test_base.h"

#ifndef SIGMOID_TEST_MODEL_PATH
#define SIGMOID_TEST_MODEL_PATH "./sigmoid_test.onnx"
#endif

class HipDNNSigmoidTest : public HipDNNTestBase {};

TEST_F(HipDNNSigmoidTest, Sigmoid) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  // Use a range that spans negative and positive values for good sigmoid coverage
  auto x = GenerateTestData(n, -3.0f, 0.025f);
  RunAndCompare(SIGMOID_TEST_MODEL_PATH, {x}, {shape}, {"X"}, "Y");
}
