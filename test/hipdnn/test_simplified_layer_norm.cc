// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "test/common/ep_test_base.h"

#ifndef SIMPLIFIED_LAYER_NORM_TEST_MODEL_PATH
#define SIMPLIFIED_LAYER_NORM_TEST_MODEL_PATH "./simplified_layer_norm_test.onnx"
#endif

#ifndef SIMPLIFIED_LAYER_NORM_RANK2_TEST_MODEL_PATH
#define SIMPLIFIED_LAYER_NORM_RANK2_TEST_MODEL_PATH \
  "./simplified_layer_norm_rank2_test.onnx"
#endif

#ifndef SIMPLIFIED_LAYER_NORM_AXIS1_TEST_MODEL_PATH
#define SIMPLIFIED_LAYER_NORM_AXIS1_TEST_MODEL_PATH \
  "./simplified_layer_norm_axis1_test.onnx"
#endif

class HipDNNSimplifiedLayerNormTest : public HipDNNTestBase {};

TEST_F(HipDNNSimplifiedLayerNormTest, Basic) {
  // X shape [1, 4, 8, 8], axis=-1 -> Scale shape [8].
  const std::vector<int64_t> x_shape = {1, 4, 8, 8};
  const std::vector<int64_t> scale_shape = {8};

  auto x = GenerateTestData(1 * 4 * 8 * 8, -2.0f, 0.016f);
  auto scale = GenerateTestData(8, 0.5f, 0.1f);

  RunAndCompare(SIMPLIFIED_LAYER_NORM_TEST_MODEL_PATH, {x, scale},
                {x_shape, scale_shape}, {"X", "Scale"}, "Y",
                /*tolerance=*/1e-3f);
}

TEST_F(HipDNNSimplifiedLayerNormTest, Rank2) {
  // X shape [8, 64], axis=-1 -> Scale shape [64].
  // Common transformer case: batch of token embeddings.
  const std::vector<int64_t> x_shape = {8, 64};
  const std::vector<int64_t> scale_shape = {64};

  auto x = GenerateTestData(8 * 64, -1.0f, 0.004f);
  auto scale = GenerateTestData(64, 0.8f, 0.005f);

  RunAndCompare(SIMPLIFIED_LAYER_NORM_RANK2_TEST_MODEL_PATH, {x, scale},
                {x_shape, scale_shape}, {"X", "Scale"}, "Y",
                /*tolerance=*/1e-3f);
}

TEST_F(HipDNNSimplifiedLayerNormTest, Axis1) {
  // X shape [2, 4, 8, 8], axis=1 -> Scale shape [4, 8, 8].
  // Exercises multi-dim norm collapse (norm_size = 4*8*8 = 256).
  const std::vector<int64_t> x_shape = {2, 4, 8, 8};
  const std::vector<int64_t> scale_shape = {4, 8, 8};

  auto x = GenerateTestData(2 * 4 * 8 * 8, -1.5f, 0.006f);
  auto scale = GenerateTestData(4 * 8 * 8, 0.3f, 0.003f);

  RunAndCompare(SIMPLIFIED_LAYER_NORM_AXIS1_TEST_MODEL_PATH, {x, scale},
                {x_shape, scale_shape}, {"X", "Scale"}, "Y",
                /*tolerance=*/1e-3f);
}
