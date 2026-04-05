// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>

#include "hipdnn_ep/core/ort_api.h"

#ifndef HIPDNN_EP_LIB_PATH
#define HIPDNN_EP_LIB_PATH "./libhipdnn_ep.so"
#endif

#ifndef MATMULNBITS_TEST_MODEL_PATH
#define MATMULNBITS_TEST_MODEL_PATH "./matmulnbits_test.onnx"
#endif

#ifndef MATMULNBITS_NO_ZP_TEST_MODEL_PATH
#define MATMULNBITS_NO_ZP_TEST_MODEL_PATH "./matmulnbits_no_zp_test.onnx"
#endif

#ifndef MATMULNBITS_LARGE_TEST_MODEL_PATH
#define MATMULNBITS_LARGE_TEST_MODEL_PATH "./matmulnbits_large_test.onnx"
#endif

class HipDNNMatMulNBitsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HipDNNMatMulNBitsTest");

    const char* lib_path = HIPDNN_EP_LIB_PATH;
    OrtStatus* status =
        Ort::GetApi().RegisterExecutionProviderLibrary(*env_, "HipDNN", lib_path);

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
    std::ifstream model_file(model_path);
    return model_file.good();
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

  // Run model with CPU EP and return output.
  std::vector<float> RunWithCpuEp(const char* model_path,
                                  const std::vector<float>& a_data,
                                  const std::vector<int64_t>& a_shape) {
    Ort::SessionOptions session_options;
    Ort::Session session(*env_, model_path, session_options);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value a_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(a_data.data()), a_data.size(),
        a_shape.data(), a_shape.size());

    const char* input_name = "A";
    const char* output_name = "Y";

    auto output_tensors =
        session.Run(Ort::RunOptions{}, &input_name, &a_tensor, 1, &output_name, 1);

    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_size = output_info.GetElementCount();

    const float* output_data = output_tensor.GetTensorData<float>();
    return std::vector<float>(output_data, output_data + output_size);
  }

  // Run model with HipDNN EP and return output.
  std::vector<float> RunWithHipDNNEp(const char* model_path,
                                     const std::vector<float>& a_data,
                                     const std::vector<int64_t>& a_shape) {
    const OrtEpDevice* device = GetHipDNNDevice();
    EXPECT_NE(device, nullptr) << "No HipDNN device found";

    Ort::SessionOptions session_options;
    OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        session_options, *env_, &device, 1, nullptr, nullptr, 0);

    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      EXPECT_TRUE(false) << "Failed to add HipDNN EP: " << error_msg;
      return {};
    }

    Ort::Session session(*env_, model_path, session_options);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value a_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(a_data.data()), a_data.size(),
        a_shape.data(), a_shape.size());

    const char* input_name = "A";
    const char* output_name = "Y";

    auto output_tensors =
        session.Run(Ort::RunOptions{}, &input_name, &a_tensor, 1, &output_name, 1);

    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_size = output_info.GetElementCount();

    const float* output_data = output_tensor.GetTensorData<float>();
    return std::vector<float>(output_data, output_data + output_size);
  }

  void CompareOutputs(const std::vector<float>& cpu_output,
                      const std::vector<float>& gpu_output,
                      float tolerance = 1e-2f) {
    ASSERT_EQ(cpu_output.size(), gpu_output.size()) << "Output size mismatch";

    float max_diff = 0.0f;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
      float diff = std::abs(cpu_output[i] - gpu_output[i]);
      max_diff = std::max(max_diff, diff);
      EXPECT_NEAR(cpu_output[i], gpu_output[i], tolerance)
          << "Mismatch at index " << i << ": CPU=" << cpu_output[i]
          << ", GPU=" << gpu_output[i];
    }
    std::cout << "Max difference between CPU and GPU: " << max_diff << std::endl;
  }

  std::unique_ptr<Ort::Env> env_;
  bool ep_available_{false};
};

TEST_F(HipDNNMatMulNBitsTest, BasicMatMulNBits) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(MATMULNBITS_TEST_MODEL_PATH))
      << "MatMulNBits test model not available at: " << MATMULNBITS_TEST_MODEL_PATH;

  // Model: A[8, 64] @ dequant(B) = Y[8, 32]
  const int64_t m = 8, k = 64;

  std::vector<float> a_data(m * k);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }

  std::cout << "Running MatMulNBits with CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(MATMULNBITS_TEST_MODEL_PATH, a_data, {m, k});

  std::cout << "Running MatMulNBits with HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(MATMULNBITS_TEST_MODEL_PATH, a_data, {m, k});

  // Use wider tolerance for quantized ops due to dequantization differences.
  CompareOutputs(cpu_output, gpu_output, 0.1f);
}

TEST_F(HipDNNMatMulNBitsTest, MatMulNBitsNoZeroPoints) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(MATMULNBITS_NO_ZP_TEST_MODEL_PATH))
      << "MatMulNBits no-zp test model not available at: "
      << MATMULNBITS_NO_ZP_TEST_MODEL_PATH;

  const int64_t m = 8, k = 64;

  std::vector<float> a_data(m * k);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }

  std::cout << "Running MatMulNBits (no zp) with CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(MATMULNBITS_NO_ZP_TEST_MODEL_PATH, a_data, {m, k});

  std::cout << "Running MatMulNBits (no zp) with HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(MATMULNBITS_NO_ZP_TEST_MODEL_PATH, a_data, {m, k});

  CompareOutputs(cpu_output, gpu_output, 0.1f);
}

TEST_F(HipDNNMatMulNBitsTest, MatMulNBitsLarger) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(MATMULNBITS_LARGE_TEST_MODEL_PATH))
      << "MatMulNBits large test model not available at: "
      << MATMULNBITS_LARGE_TEST_MODEL_PATH;

  // Model: A[16, 128] @ dequant(B) = Y[16, 64]
  const int64_t m = 16, k = 128;

  std::vector<float> a_data(m * k);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }

  std::cout << "Running MatMulNBits (larger) with CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(MATMULNBITS_LARGE_TEST_MODEL_PATH, a_data, {m, k});

  std::cout << "Running MatMulNBits (larger) with HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(MATMULNBITS_LARGE_TEST_MODEL_PATH, a_data, {m, k});

  CompareOutputs(cpu_output, gpu_output, 0.1f);
}
