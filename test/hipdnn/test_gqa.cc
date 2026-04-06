// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "sdpa_test_base.h"

#ifndef GQA_MHA_TEST_MODEL_PATH
#define GQA_MHA_TEST_MODEL_PATH "./gqa_mha_test.onnx"
#endif

class HipDNNGqaTest : public HipDNNSdpaTestBase {};

// MHA baseline via GQA op: 8 query heads, 8 KV heads (equal).
// This exercises the GQA code path with standard MHA semantics.
// B=2, S_q=16, S_kv=16, H_q=8, H_kv=8, D=64
// Q hidden = KV hidden = 512
TEST_F(HipDNNGqaTest, MhaBaseline) {
  const std::vector<int64_t> q_shape = {2, 16, 512};
  const std::vector<int64_t> k_shape = {2, 16, 512};
  const std::vector<int64_t> v_shape = {2, 16, 512};
  size_t q_n = 2 * 16 * 512;
  size_t k_n = 2 * 16 * 512;
  size_t v_n = 2 * 16 * 512;

  auto q = GenerateTestData(q_n, -0.2f, 0.00008f);
  auto k = GenerateTestData(k_n, 0.2f, 0.00008f);
  auto v = GenerateTestData(v_n, 0.0f, 0.0001f);

  RunAndCompare(GQA_MHA_TEST_MODEL_PATH, q_shape, q, k_shape, k, v_shape, v);
}
