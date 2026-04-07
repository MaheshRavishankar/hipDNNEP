// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#ifndef HIPDNN_EP_TEST_COMMON_EP_TEST_BASE_H_
#define HIPDNN_EP_TEST_COMMON_EP_TEST_BASE_H_

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "hipdnn_ep/core/ort_api.h"

#ifndef HIPDNN_EP_LIB_PATH
#define HIPDNN_EP_LIB_PATH "./libhipdnn_ep.so"
#endif

// Base test fixture for all hipDNN EP tests that run ONNX models.
// Handles ORT initialization, EP registration, device discovery,
// CPU vs GPU inference, and output comparison.
class HipDNNTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                      "HipDNNTest");

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

  bool IsModelAvailable(const char* model_path) {
    std::ifstream f(model_path);
    return f.good();
  }

  const OrtEpDevice* GetHipDNNDevice() {
    std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
    for (const auto& device : devices) {
      if (std::string(device.EpName()) == "HipDNN") {
        return static_cast<const OrtEpDevice*>(device);
      }
    }
    return nullptr;
  }

  // Run a model on the CPU EP and return the first output as a flat vector.
  std::vector<float> RunCpu(
      const char* model_path,
      const std::vector<std::vector<float>>& inputs,
      const std::vector<std::vector<int64_t>>& shapes,
      const std::vector<const char*>& input_names,
      const char* output_name) {
    Ort::SessionOptions opts;
    Ort::Session session(*env_, model_path, opts);

    auto mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> tensors;
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensors.push_back(Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(inputs[i].data()), inputs[i].size(),
          shapes[i].data(), shapes[i].size()));
    }

    auto outputs = session.Run(Ort::RunOptions{}, input_names.data(),
                               tensors.data(), tensors.size(), &output_name,
                               1);
    EXPECT_EQ(outputs.size(), 1u);
    size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* data = outputs[0].GetTensorData<float>();
    return {data, data + n};
  }

  // Run a model on the HipDNN EP and return the first output as a flat vector.
  std::vector<float> RunHipDNN(
      const char* model_path,
      const std::vector<std::vector<float>>& inputs,
      const std::vector<std::vector<int64_t>>& shapes,
      const std::vector<const char*>& input_names,
      const char* output_name) {
    const OrtEpDevice* device = GetHipDNNDevice();
    EXPECT_NE(device, nullptr) << "No HipDNN device found";
    if (device == nullptr) return {};

    Ort::SessionOptions opts;
    OrtStatus* status =
        Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
            opts, *env_, &device, 1, nullptr, nullptr, 0);
    if (status != nullptr) {
      std::string msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      ADD_FAILURE() << "Failed to add HipDNN EP: " << msg;
      return {};
    }

    Ort::Session session(*env_, model_path, opts);

    auto mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> tensors;
    for (size_t i = 0; i < inputs.size(); ++i) {
      tensors.push_back(Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(inputs[i].data()), inputs[i].size(),
          shapes[i].data(), shapes[i].size()));
    }

    auto outputs = session.Run(Ort::RunOptions{}, input_names.data(),
                               tensors.data(), tensors.size(), &output_name,
                               1);
    EXPECT_EQ(outputs.size(), 1u);
    size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* data = outputs[0].GetTensorData<float>();
    return {data, data + n};
  }

  // Compare CPU and GPU outputs element-wise.
  void CompareOutputs(const std::vector<float>& cpu_output,
                      const std::vector<float>& gpu_output,
                      float tolerance = 1e-4f) {
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

  // All-in-one: run on CPU and GPU, then compare.
  void RunAndCompare(
      const char* model_path,
      const std::vector<std::vector<float>>& inputs,
      const std::vector<std::vector<int64_t>>& shapes,
      const std::vector<const char*>& input_names,
      const char* output_name,
      float tolerance = 1e-4f) {
    ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
    ASSERT_TRUE(IsModelAvailable(model_path))
        << "Model not available at: " << model_path;

    auto cpu_output = RunCpu(model_path, inputs, shapes, input_names,
                             output_name);
    auto gpu_output = RunHipDNN(model_path, inputs, shapes, input_names,
                                output_name);
    CompareOutputs(cpu_output, gpu_output, tolerance);
  }

  std::unique_ptr<Ort::Env> env_;
  bool ep_available_{false};
};

// Helper to generate deterministic test data with a linear ramp.
static inline std::vector<float> GenerateTestData(size_t n, float base,
                                                  float step) {
  std::vector<float> data(n);
  for (size_t i = 0; i < n; ++i) {
    data[i] = base + static_cast<float>(i) * step;
  }
  return data;
}

#endif  // HIPDNN_EP_TEST_COMMON_EP_TEST_BASE_H_
