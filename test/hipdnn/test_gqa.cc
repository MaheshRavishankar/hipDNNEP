// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <vector>

#include "hipdnn_ep/core/ort_api.h"

#ifndef HIPDNN_EP_LIB_PATH
#define HIPDNN_EP_LIB_PATH "./libhipdnn_ep.so"
#endif

#ifndef GQA_TEST_MODEL_PATH
#define GQA_TEST_MODEL_PATH "./gqa_test.onnx"
#endif

#ifndef GQA_MQA_TEST_MODEL_PATH
#define GQA_MQA_TEST_MODEL_PATH "./gqa_mqa_test.onnx"
#endif

#ifndef GQA_MHA_TEST_MODEL_PATH
#define GQA_MHA_TEST_MODEL_PATH "./gqa_mha_test.onnx"
#endif

class HipDNNGqaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      "HipDNNGqaTest");

    // Register EP
    const char* lib_path = HIPDNN_EP_LIB_PATH;
    OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
        *env_, "HipDNN", lib_path);

    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      ep_available_ = false;
      std::cout << "EP not available: " << error_msg << std::endl;
    } else {
      ep_available_ = true;
    }
  }

  void TearDown() override { env_.reset(); }

  // Run the GQA model on CPU and GPU, then compare results.
  void RunAndCompare(
      const char* model_path,
      const std::vector<int64_t>& q_shape,
      const std::vector<float>& q_data,
      const std::vector<int64_t>& k_shape,
      const std::vector<float>& k_data,
      const std::vector<int64_t>& v_shape,
      const std::vector<float>& v_data,
      float tolerance = 1e-3f) {
    ASSERT_TRUE(ep_available_) << "HipDNN EP not available";

    std::ifstream model_file(model_path);
    ASSERT_TRUE(model_file.good())
        << "Model not available at: " << model_path;

    // --- CPU reference ---
    std::vector<float> cpu_output;
    {
      Ort::SessionOptions opts;
      Ort::Session session(*env_, model_path, opts);

      auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                            OrtMemTypeDefault);
      Ort::Value q_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(q_data.data()), q_data.size(),
          q_shape.data(), q_shape.size());
      Ort::Value k_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(k_data.data()), k_data.size(),
          k_shape.data(), k_shape.size());
      Ort::Value v_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(v_data.data()), v_data.size(),
          v_shape.data(), v_shape.size());

      const char* input_names[] = {"query", "key", "value"};
      const char* output_names[] = {"output"};
      Ort::Value inputs[] = {std::move(q_tensor), std::move(k_tensor),
                             std::move(v_tensor)};

      auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs,
                                 3, output_names, 1);
      ASSERT_EQ(outputs.size(), 1u);
      size_t out_size =
          outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
      const float* data = outputs[0].GetTensorData<float>();
      cpu_output.assign(data, data + out_size);
    }

    // --- GPU (HipDNN EP) ---
    std::vector<float> gpu_output;
    {
      std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
      ASSERT_FALSE(devices.empty()) << "No EP devices found";

      const OrtEpDevice* hipdnn_device = nullptr;
      for (const auto& device : devices) {
        if (device.EpName() == std::string("HipDNN")) {
          hipdnn_device = static_cast<const OrtEpDevice*>(device);
          break;
        }
      }
      ASSERT_NE(hipdnn_device, nullptr) << "No HipDNN device found";

      Ort::SessionOptions opts;
      OrtStatus* status =
          Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
              opts, *env_, &hipdnn_device, 1, nullptr, nullptr, 0);
      if (status != nullptr) {
        std::string msg = Ort::GetApi().GetErrorMessage(status);
        Ort::GetApi().ReleaseStatus(status);
        FAIL() << "Failed to add HipDNN EP: " << msg;
      }

      try {
        Ort::Session session(*env_, model_path, opts);

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                              OrtMemTypeDefault);
        Ort::Value q_tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(q_data.data()), q_data.size(),
            q_shape.data(), q_shape.size());
        Ort::Value k_tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(k_data.data()), k_data.size(),
            k_shape.data(), k_shape.size());
        Ort::Value v_tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(v_data.data()), v_data.size(),
            v_shape.data(), v_shape.size());

        const char* input_names[] = {"query", "key", "value"};
        const char* output_names[] = {"output"};
        Ort::Value inputs[] = {std::move(q_tensor), std::move(k_tensor),
                               std::move(v_tensor)};

        auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs,
                                   3, output_names, 1);
        ASSERT_EQ(outputs.size(), 1u);
        size_t out_size =
            outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* data = outputs[0].GetTensorData<float>();
        gpu_output.assign(data, data + out_size);
      } catch (const std::exception& e) {
        std::string msg = e.what();
        if (msg.find("No engine configurations") != std::string::npos ||
            msg.find("create_execution_plans") != std::string::npos) {
          GTEST_SKIP() << "SDPA engine not available in current hipDNN build: "
                       << msg;
        }
        throw;
      }
    }

    // --- Compare ---
    ASSERT_EQ(cpu_output.size(), gpu_output.size()) << "Output size mismatch";

    float max_diff = 0.0f;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
      float diff = std::abs(cpu_output[i] - gpu_output[i]);
      max_diff = std::max(max_diff, diff);
      EXPECT_NEAR(cpu_output[i], gpu_output[i], tolerance)
          << "Mismatch at index " << i << ": CPU=" << cpu_output[i]
          << ", GPU=" << gpu_output[i];
    }
    std::cout << "Max difference between CPU and GPU: " << max_diff
              << std::endl;
  }

  std::unique_ptr<Ort::Env> env_;
  bool ep_available_{false};
};

// Helper to generate deterministic test data with small values
// to keep attention weights reasonable.
static std::vector<float> GenerateTestData(size_t n, float base, float step) {
  std::vector<float> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = base + static_cast<float>(i) * step;
  }
  return data;
}

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
