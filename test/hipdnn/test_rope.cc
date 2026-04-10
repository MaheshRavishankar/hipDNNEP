// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "hipdnn_ep/core/ort_api.h"

#ifndef HIPDNN_EP_LIB_PATH
#define HIPDNN_EP_LIB_PATH "./libhipdnn_ep.so"
#endif

#ifndef ROPE_TEST_MODEL_PATH
#define ROPE_TEST_MODEL_PATH "./rope_test.onnx"
#endif

class HipDNNRoPETest : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      "HipDNNRoPETest");

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

  // Compute cos/sin caches for RoPE.
  // Returns vectors of shape [seq_len * half_dim].
  static void ComputeRoPECaches(int64_t seq_len, int64_t head_size,
                                std::vector<float>& cos_out,
                                std::vector<float>& sin_out,
                                float base = 10000.0f) {
    int64_t half = head_size / 2;
    cos_out.resize(seq_len * half);
    sin_out.resize(seq_len * half);
    for (int64_t s = 0; s < seq_len; ++s) {
      for (int64_t d = 0; d < half; ++d) {
        float inv_freq =
            1.0f / std::pow(base, static_cast<float>(d) / static_cast<float>(half));
        float angle = static_cast<float>(s) * inv_freq;
        cos_out[s * half + d] = std::cos(angle);
        sin_out[s * half + d] = std::sin(angle);
      }
    }
  }

  // Run a RoPE model on CPU and GPU, then compare results.
  void RunAndCompare(
      const char* model_path,
      const std::vector<int64_t>& input_shape,
      const std::vector<float>& input_data,
      const std::vector<int64_t>& pos_shape,
      const std::vector<int64_t>& pos_data,
      const std::vector<int64_t>& cos_shape,
      const std::vector<float>& cos_data,
      const std::vector<int64_t>& sin_shape,
      const std::vector<float>& sin_data,
      float tolerance = 1e-4f) {
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
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(input_data.data()), input_data.size(),
          input_shape.data(), input_shape.size());
      Ort::Value pos_tensor = Ort::Value::CreateTensor<int64_t>(
          mem, const_cast<int64_t*>(pos_data.data()), pos_data.size(),
          pos_shape.data(), pos_shape.size());
      Ort::Value cos_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(cos_data.data()), cos_data.size(),
          cos_shape.data(), cos_shape.size());
      Ort::Value sin_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(sin_data.data()), sin_data.size(),
          sin_shape.data(), sin_shape.size());

      const char* input_names[] = {"input", "position_ids", "cos_cache",
                                   "sin_cache"};
      const char* output_names[] = {"output"};
      Ort::Value inputs[] = {std::move(input_tensor), std::move(pos_tensor),
                             std::move(cos_tensor), std::move(sin_tensor)};

      auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs,
                                 4, output_names, 1);
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
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(input_data.data()), input_data.size(),
            input_shape.data(), input_shape.size());
        Ort::Value pos_tensor = Ort::Value::CreateTensor<int64_t>(
            mem, const_cast<int64_t*>(pos_data.data()), pos_data.size(),
            pos_shape.data(), pos_shape.size());
        Ort::Value cos_tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(cos_data.data()), cos_data.size(),
            cos_shape.data(), cos_shape.size());
        Ort::Value sin_tensor = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(sin_data.data()), sin_data.size(),
            sin_shape.data(), sin_shape.size());

        const char* input_names[] = {"input", "position_ids", "cos_cache",
                                     "sin_cache"};
        const char* output_names[] = {"output"};
        Ort::Value inputs[] = {std::move(input_tensor), std::move(pos_tensor),
                               std::move(cos_tensor), std::move(sin_tensor)};

        auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs,
                                   4, output_names, 1);
        ASSERT_EQ(outputs.size(), 1u);
        size_t out_size =
            outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* data = outputs[0].GetTensorData<float>();
        gpu_output.assign(data, data + out_size);
      } catch (const std::exception& e) {
        std::string msg = e.what();
        if (msg.find("No engine configurations") != std::string::npos ||
            msg.find("create_execution_plans") != std::string::npos ||
            msg.find("Unsupported node type") != std::string::npos) {
          GTEST_SKIP()
              << "RoPE custom op engine not available in current hipDNN build: "
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

// Basic RoPE: B=2, H=4, S=16, D=64
TEST_F(HipDNNRoPETest, BasicRoPE) {
  const int64_t B = 2, H = 4, S = 16, D = 64;
  const int64_t half_d = D / 2;

  const std::vector<int64_t> input_shape = {B, H, S, D};
  const size_t input_n = B * H * S * D;

  // Generate deterministic input data.
  std::vector<float> input_data(input_n);
  for (size_t i = 0; i < input_n; ++i) {
    input_data[i] = -0.5f + static_cast<float>(i) * 0.0001f;
  }

  // Sequential position IDs: [B, S] — same positions for each batch.
  const std::vector<int64_t> pos_shape = {B, S};
  std::vector<int64_t> pos_data(B * S);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < S; ++i) {
      pos_data[b * S + i] = i;
    }
  }

  // Compute cos/sin caches.
  const std::vector<int64_t> cache_shape = {S, half_d};
  std::vector<float> cos_data, sin_data;
  ComputeRoPECaches(S, D, cos_data, sin_data);

  RunAndCompare(ROPE_TEST_MODEL_PATH, input_shape, input_data,
                pos_shape, pos_data, cache_shape, cos_data,
                cache_shape, sin_data);
}
