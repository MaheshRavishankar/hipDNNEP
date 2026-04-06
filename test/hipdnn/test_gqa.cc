// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "sdpa_test_base.h"

#ifndef GQA_TEST_MODEL_PATH
#define GQA_TEST_MODEL_PATH "./gqa_test.onnx"
#endif

#ifndef GQA_MQA_TEST_MODEL_PATH
#define GQA_MQA_TEST_MODEL_PATH "./gqa_mqa_test.onnx"
#endif

#ifndef GQA_MHA_TEST_MODEL_PATH
#define GQA_MHA_TEST_MODEL_PATH "./gqa_mha_test.onnx"
#endif

#ifndef GQA_LONG_SEQ_TEST_MODEL_PATH
#define GQA_LONG_SEQ_TEST_MODEL_PATH "./gqa_long_seq_test.onnx"
#endif

class HipDNNGqaTest : public HipDNNSdpaTestBase {};

// Standard GQA: 8 query heads, 2 KV heads (each KV head shared across 4 Q heads).
// B=2, S_q=16, S_kv=16, H_q=8, H_kv=2, D=64
// Q hidden = 8*64 = 512, KV hidden = 2*64 = 128
TEST_F(HipDNNGqaTest, StandardGqa) {
  const std::vector<int64_t> q_shape = {2, 16, 512};
  const std::vector<int64_t> k_shape = {2, 16, 128};
  const std::vector<int64_t> v_shape = {2, 16, 128};
  size_t q_n = 2 * 16 * 512;
  size_t k_n = 2 * 16 * 128;
  size_t v_n = 2 * 16 * 128;

  auto q = GenerateTestData(q_n, -0.5f, 0.0001f);
  auto k = GenerateTestData(k_n, 0.0f, 0.0003f);
  auto v = GenerateTestData(v_n, 0.1f, 0.0002f);

  RunAndCompare(GQA_TEST_MODEL_PATH, q_shape, q, k_shape, k, v_shape, v);
}

// Multi-Query Attention (MQA): 8 query heads, 1 KV head.
// B=2, S_q=16, S_kv=16, H_q=8, H_kv=1, D=64
// Q hidden = 512, KV hidden = 64
TEST_F(HipDNNGqaTest, MultiQueryAttention) {
  const std::vector<int64_t> q_shape = {2, 16, 512};
  const std::vector<int64_t> k_shape = {2, 16, 64};
  const std::vector<int64_t> v_shape = {2, 16, 64};
  size_t q_n = 2 * 16 * 512;
  size_t k_n = 2 * 16 * 64;
  size_t v_n = 2 * 16 * 64;

  auto q = GenerateTestData(q_n, -0.3f, 0.0001f);
  auto k = GenerateTestData(k_n, 0.1f, 0.0005f);
  auto v = GenerateTestData(v_n, -0.1f, 0.0004f);

  RunAndCompare(GQA_MQA_TEST_MODEL_PATH, q_shape, q, k_shape, k, v_shape, v);
}

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

// Longer sequence GQA: S=32 with grouped heads (exercises larger attention
// matrices than the standard S=16 tests).
// B=2, S=32, H_q=8, H_kv=2, D=64
// Q hidden = 512, KV hidden = 128
TEST_F(HipDNNGqaTest, LongSequenceGqa) {
  const std::vector<int64_t> q_shape = {2, 32, 512};
  const std::vector<int64_t> k_shape = {2, 32, 128};
  const std::vector<int64_t> v_shape = {2, 32, 128};
  size_t q_n = 2 * 32 * 512;
  size_t k_n = 2 * 32 * 128;
  size_t v_n = 2 * 32 * 128;

  auto q = GenerateTestData(q_n, -0.4f, 0.0001f);
  auto k = GenerateTestData(k_n, 0.1f, 0.00015f);
  auto v = GenerateTestData(v_n, -0.2f, 0.0001f);

  RunAndCompare(GQA_LONG_SEQ_TEST_MODEL_PATH, q_shape, q, k_shape, k,
                v_shape, v);
}
