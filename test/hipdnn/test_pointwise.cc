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

class HipDNNPointwiseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      "HipDNNPointwiseTest");

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

  // Run a two-input, one-output model on CPU and GPU, then compare.
  void RunAndCompare(
      const char* model_path,
      const std::vector<int64_t>& shape,
      const std::vector<float>& a_data,
      const std::vector<float>& b_data,
      float tolerance = 1e-4f) {
    ASSERT_TRUE(ep_available_) << "HipDNN EP not available";

    std::ifstream model_file(model_path);
    ASSERT_TRUE(model_file.good())
        << "Model not available at: " << model_path;

    size_t num_elements = a_data.size();

    // --- CPU reference ---
    std::vector<float> cpu_output;
    {
      Ort::SessionOptions opts;
      Ort::Session session(*env_, model_path, opts);

      auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                            OrtMemTypeDefault);
      Ort::Value a_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(a_data.data()), num_elements,
          shape.data(), shape.size());
      Ort::Value b_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(b_data.data()), num_elements,
          shape.data(), shape.size());

      const char* input_names[] = {"A", "B"};
      const char* output_names[] = {"Y"};
      Ort::Value inputs[] = {std::move(a_tensor), std::move(b_tensor)};

      auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs,
                                 2, output_names, 1);
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

      Ort::Session session(*env_, model_path, opts);

      auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                            OrtMemTypeDefault);
      Ort::Value a_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(a_data.data()), num_elements,
          shape.data(), shape.size());
      Ort::Value b_tensor = Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(b_data.data()), num_elements,
          shape.data(), shape.size());

      const char* input_names[] = {"A", "B"};
      const char* output_names[] = {"Y"};
      Ort::Value inputs[] = {std::move(a_tensor), std::move(b_tensor)};

      auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs,
                                 2, output_names, 1);
      ASSERT_EQ(outputs.size(), 1u);
      size_t out_size =
          outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
      const float* data = outputs[0].GetTensorData<float>();
      gpu_output.assign(data, data + out_size);
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

// Helper to generate deterministic test data
static std::vector<float> GenerateTestData(size_t n, float base, float step) {
  std::vector<float> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = base + static_cast<float>(i) * step;
  }
  return data;
}

TEST_F(HipDNNPointwiseTest, Mul) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 0.1f, 0.01f);
  auto b = GenerateTestData(n, 1.0f, -0.003f);
  RunAndCompare(MUL_TEST_MODEL_PATH, shape, a, b);
}

TEST_F(HipDNNPointwiseTest, Sub) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 5.0f, -0.01f);
  auto b = GenerateTestData(n, 0.5f, 0.02f);
  RunAndCompare(SUB_TEST_MODEL_PATH, shape, a, b);
}

TEST_F(HipDNNPointwiseTest, Add) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, -1.0f, 0.01f);
  auto b = GenerateTestData(n, 2.0f, -0.005f);
  RunAndCompare(ADD_TEST_MODEL_PATH, shape, a, b);
}

TEST_F(HipDNNPointwiseTest, Div) {
  const std::vector<int64_t> shape = {1, 4, 8, 8};
  size_t n = 1 * 4 * 8 * 8;
  auto a = GenerateTestData(n, 1.0f, 0.05f);
  // Avoid zero denominators
  auto b = GenerateTestData(n, 1.0f, 0.01f);
  RunAndCompare(DIV_TEST_MODEL_PATH, shape, a, b, 1e-3f);
}
