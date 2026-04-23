// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "test/common/ep_test_base.h"

class HipDNNReductionTest : public HipDNNTestBase {
 protected:
  // Generate a deterministic ramp that straddles zero so min/max/sum all
  // produce distinguishable results.
  std::vector<float> MakeInput(size_t n) {
    return GenerateTestData(n, -1.0f, 0.01f);
  }
};

// ReduceSum: axes are passed as a constant input tensor in opset 13.
TEST_F(HipDNNReductionTest, ReduceSumAxesInput) {
  const std::vector<int64_t> shape = {2, 4, 8, 8};
  auto x = MakeInput(2 * 4 * 8 * 8);
  RunAndCompare(REDUCESUM_TEST_MODEL_PATH, {x}, {shape}, {"X"}, "Y");
}

// ReduceMean: axes is an attribute in opset 13.
TEST_F(HipDNNReductionTest, ReduceMean) {
  const std::vector<int64_t> shape = {2, 4, 8, 8};
  auto x = MakeInput(2 * 4 * 8 * 8);
  RunAndCompare(REDUCEMEAN_TEST_MODEL_PATH, {x}, {shape}, {"X"}, "Y");
}

TEST_F(HipDNNReductionTest, ReduceMax) {
  const std::vector<int64_t> shape = {2, 4, 8, 8};
  auto x = MakeInput(2 * 4 * 8 * 8);
  RunAndCompare(REDUCEMAX_TEST_MODEL_PATH, {x}, {shape}, {"X"}, "Y");
}

TEST_F(HipDNNReductionTest, ReduceMin) {
  const std::vector<int64_t> shape = {2, 4, 8, 8};
  auto x = MakeInput(2 * 4 * 8 * 8);
  RunAndCompare(REDUCEMIN_TEST_MODEL_PATH, {x}, {shape}, {"X"}, "Y");
}

// Reduce every axis to a scalar (with keepdims=1 so the result is [1,1,1,1]).
TEST_F(HipDNNReductionTest, ReduceSumAllAxes) {
  const std::vector<int64_t> shape = {2, 3, 4, 5};
  auto x = MakeInput(2 * 3 * 4 * 5);
  RunAndCompare(REDUCESUM_ALL_TEST_MODEL_PATH, {x}, {shape}, {"X"}, "Y");
}

// Reduce a single axis in the middle of the tensor.
TEST_F(HipDNNReductionTest, ReduceSumSingleAxis) {
  const std::vector<int64_t> shape = {2, 4, 8, 8};
  auto x = MakeInput(2 * 4 * 8 * 8);
  RunAndCompare(REDUCESUM_SINGLE_AXIS_TEST_MODEL_PATH, {x}, {shape}, {"X"},
                "Y");
}
