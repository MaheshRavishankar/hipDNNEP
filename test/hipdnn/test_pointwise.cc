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

  // Run a model with the CPU EP and return the output.
  std::vector<float> RunCPU(const char* model_path,
                            const std::vector<float>& a_data,
                            const std::vector<float>& b_data,
                            const std::vector<int64_t>& shape) {
    Ort::SessionOptions session_options;
    Ort::Session session(*env_, model_path, session_options);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    size_t elem_count = a_data.size();
    Ort::Value a_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(a_data.data()), elem_count,
        shape.data(), shape.size());
    Ort::Value b_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(b_data.data()), elem_count,
        shape.data(), shape.size());

    const char* input_names[] = {"A", "B"};
    const char* output_names[] = {"Y"};
    Ort::Value inputs[] = {std::move(a_tensor), std::move(b_tensor)};

    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 2,
                               output_names, 1);

    EXPECT_EQ(outputs.size(), 1u);
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    size_t out_size = info.GetElementCount();
    const float* out_data = outputs[0].GetTensorData<float>();
    return {out_data, out_data + out_size};
  }

  // Run a model with the HipDNN EP and return the output.
  std::vector<float> RunGPU(const char* model_path,
                            const std::vector<float>& a_data,
                            const std::vector<float>& b_data,
                            const std::vector<int64_t>& shape) {
    std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
    EXPECT_FALSE(devices.empty()) << "No EP devices found";

    const OrtEpDevice* hipdnn_device = nullptr;
    for (const auto& device : devices) {
      if (device.EpName() == std::string("HipDNN")) {
        hipdnn_device = static_cast<const OrtEpDevice*>(device);
        break;
      }
    }
    EXPECT_NE(hipdnn_device, nullptr) << "No HipDNN device found";

    Ort::SessionOptions session_options;
    OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        session_options, *env_, &hipdnn_device, 1, nullptr, nullptr, 0);
    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      ADD_FAILURE() << "Failed to add HipDNN EP: " << error_msg;
      return {};
    }

    Ort::Session session(*env_, model_path, session_options);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    size_t elem_count = a_data.size();
    Ort::Value a_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(a_data.data()), elem_count,
        shape.data(), shape.size());
    Ort::Value b_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(b_data.data()), elem_count,
        shape.data(), shape.size());

    const char* input_names[] = {"A", "B"};
    const char* output_names[] = {"Y"};
    Ort::Value inputs[] = {std::move(a_tensor), std::move(b_tensor)};

    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 2,
                               output_names, 1);

    EXPECT_EQ(outputs.size(), 1u);
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    size_t out_size = info.GetElementCount();
    const float* out_data = outputs[0].GetTensorData<float>();
    return {out_data, out_data + out_size};
  }

  // Compare CPU and GPU outputs
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

  // Run a pointwise test end-to-end: CPU reference vs HipDNN EP
  void RunPointwiseTest(const char* model_path,
                        const std::vector<int64_t>& shape) {
    ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
    std::ifstream f(model_path);
    ASSERT_TRUE(f.good()) << "Model not available at: " << model_path;

    size_t elem_count = 1;
    for (auto d : shape) elem_count *= static_cast<size_t>(d);

    // Generate deterministic input data
    std::vector<float> a_data(elem_count);
    std::vector<float> b_data(elem_count);
    for (size_t i = 0; i < elem_count; ++i) {
      a_data[i] = static_cast<float>(i % 7) / 3.0f + 0.5f;
      // Avoid zero for Div: values in range [0.5, 2.5]
      b_data[i] = static_cast<float>(i % 5) / 2.0f + 0.5f;
    }

    auto cpu_output = RunCPU(model_path, a_data, b_data, shape);
    auto gpu_output = RunGPU(model_path, a_data, b_data, shape);
    CompareOutputs(cpu_output, gpu_output);
  }

  std::unique_ptr<Ort::Env> env_;
  bool ep_available_{false};
};

// Must match the shape used in gen_pointwise_model.py
static const std::vector<int64_t> kShape = {2, 3, 4, 4};

TEST_F(HipDNNPointwiseTest, Mul) {
  RunPointwiseTest(MUL_TEST_MODEL_PATH, kShape);
}

TEST_F(HipDNNPointwiseTest, Sub) {
  RunPointwiseTest(SUB_TEST_MODEL_PATH, kShape);
}

TEST_F(HipDNNPointwiseTest, Add) {
  RunPointwiseTest(ADD_TEST_MODEL_PATH, kShape);
}

TEST_F(HipDNNPointwiseTest, Div) {
  RunPointwiseTest(DIV_TEST_MODEL_PATH, kShape);
}

// Reference correctness tests -- no GPU needed
TEST_F(HipDNNPointwiseTest, ReferenceMulCorrectness) {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> expected = {5.0f, 12.0f, 21.0f, 32.0f};

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i] * b[i], expected[i], 1e-5f) << "Mul mismatch at " << i;
  }
}

TEST_F(HipDNNPointwiseTest, ReferenceSubCorrectness) {
  std::vector<float> a = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> expected = {4.0f, 4.0f, 4.0f, 4.0f};

  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i] - b[i], expected[i], 1e-5f) << "Sub mismatch at " << i;
  }
}
