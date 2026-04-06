// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "sdpa_test_base.h"

#ifndef MHA_TEST_MODEL_PATH
#define MHA_TEST_MODEL_PATH "./mha_test.onnx"
#endif

#ifndef MHA_CAUSAL_TEST_MODEL_PATH
#define MHA_CAUSAL_TEST_MODEL_PATH "./mha_causal_test.onnx"
#endif

#ifndef MHA_CROSS_TEST_MODEL_PATH
#define MHA_CROSS_TEST_MODEL_PATH "./mha_cross_test.onnx"
#endif

class HipDNNMhaTest : public HipDNNSdpaTestBase {};

// Basic SDPA: Q, K, V with default scale, no causal mask.
// B=2, S_q=16, S_kv=16, H=4, D=64, hidden=256
TEST_F(HipDNNMhaTest, BasicSdpa) {
  const std::vector<int64_t> q_shape = {2, 16, 256};
  const std::vector<int64_t> k_shape = {2, 16, 256};
  const std::vector<int64_t> v_shape = {2, 16, 256};
  size_t q_n = 2 * 16 * 256;
  size_t k_n = 2 * 16 * 256;
  size_t v_n = 2 * 16 * 256;

  auto q = GenerateTestData(q_n, -0.5f, 0.0002f);
  auto k = GenerateTestData(k_n, 0.0f, 0.0002f);
  auto v = GenerateTestData(v_n, 0.1f, 0.0001f);

  RunAndCompare(MHA_TEST_MODEL_PATH, q_shape, q, k_shape, k, v_shape, v);
}

// SDPA with causal masking (unidirectional=1).
TEST_F(HipDNNMhaTest, CausalSdpa) {
  const std::vector<int64_t> q_shape = {2, 16, 256};
  const std::vector<int64_t> k_shape = {2, 16, 256};
  const std::vector<int64_t> v_shape = {2, 16, 256};
  size_t q_n = 2 * 16 * 256;
  size_t k_n = 2 * 16 * 256;
  size_t v_n = 2 * 16 * 256;

  auto q = GenerateTestData(q_n, -0.3f, 0.0002f);
  auto k = GenerateTestData(k_n, 0.1f, 0.0002f);
  auto v = GenerateTestData(v_n, -0.1f, 0.0001f);

  RunAndCompare(MHA_CAUSAL_TEST_MODEL_PATH, q_shape, q, k_shape, k,
                v_shape, v);
}

// Cross-attention: S_q != S_kv.
// B=2, S_q=16, S_kv=32, H=4, D=64, hidden=256
TEST_F(HipDNNMhaTest, CrossAttentionSdpa) {
  const std::vector<int64_t> q_shape = {2, 16, 256};
  const std::vector<int64_t> k_shape = {2, 32, 256};
  const std::vector<int64_t> v_shape = {2, 32, 256};
  size_t q_n = 2 * 16 * 256;
  size_t k_n = 2 * 32 * 256;
  size_t v_n = 2 * 32 * 256;

  auto q = GenerateTestData(q_n, -0.4f, 0.0002f);
  auto k = GenerateTestData(k_n, 0.1f, 0.0001f);
  auto v = GenerateTestData(v_n, -0.2f, 0.00008f);

  RunAndCompare(MHA_CROSS_TEST_MODEL_PATH, q_shape, q, k_shape, k,
                v_shape, v);
}
