// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#ifndef HIPDNN_EP_LIB_PATH
#define HIPDNN_EP_LIB_PATH "./libhipdnn_ep.so"
#endif

#ifndef MATMUL_TEST_MODEL_PATH
#define MATMUL_TEST_MODEL_PATH "./matmul_test.onnx"
#endif

#ifndef GEMM_TEST_MODEL_PATH
#define GEMM_TEST_MODEL_PATH "./gemm_test.onnx"
#endif

#ifndef GEMM_BIAS_TEST_MODEL_PATH
#define GEMM_BIAS_TEST_MODEL_PATH "./gemm_bias_test.onnx"
#endif

#ifndef GEMM_TRANS_A_TEST_MODEL_PATH
#define GEMM_TRANS_A_TEST_MODEL_PATH "./gemm_trans_a_test.onnx"
#endif

#ifndef GEMM_TRANS_B_TEST_MODEL_PATH
#define GEMM_TRANS_B_TEST_MODEL_PATH "./gemm_trans_b_test.onnx"
#endif

#ifndef GEMM_SCALED_TEST_MODEL_PATH
#define GEMM_SCALED_TEST_MODEL_PATH "./gemm_scaled_test.onnx"
#endif

class HipDNNMatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HipDNNMatMulTest");

    // Register EP
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

  // Run model with CPU EP and return output
  std::vector<float> RunWithCpuEp(const char* model_path,
                                  const std::vector<std::vector<float>>& inputs,
                                  const std::vector<std::vector<int64_t>>& input_shapes,
                                  const std::vector<const char*>& input_names,
                                  const char* output_name) {
    Ort::SessionOptions session_options;
    Ort::Session session(*env_, model_path, session_options);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory_info, const_cast<float*>(inputs[i].data()), inputs[i].size(),
          input_shapes[i].data(), input_shapes[i].size()));
    }

    std::vector<const OrtValue*> input_ptrs;
    for (auto& tensor : input_tensors) {
      input_ptrs.push_back(tensor);
    }

    auto output_tensors =
        session.Run(Ort::RunOptions{}, input_names.data(), input_tensors.data(),
                    input_tensors.size(), &output_name, 1);

    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_size = output_info.GetElementCount();

    const float* output_data = output_tensor.GetTensorData<float>();
    return std::vector<float>(output_data, output_data + output_size);
  }

  // Run model with HipDNN EP and return output
  std::vector<float> RunWithHipDNNEp(const char* model_path,
                                     const std::vector<std::vector<float>>& inputs,
                                     const std::vector<std::vector<int64_t>>& input_shapes,
                                     const std::vector<const char*>& input_names,
                                     const char* output_name) {
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

    std::vector<Ort::Value> input_tensors;
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory_info, const_cast<float*>(inputs[i].data()), inputs[i].size(),
          input_shapes[i].data(), input_shapes[i].size()));
    }

    auto output_tensors =
        session.Run(Ort::RunOptions{}, input_names.data(), input_tensors.data(),
                    input_tensors.size(), &output_name, 1);

    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_size = output_info.GetElementCount();

    const float* output_data = output_tensor.GetTensorData<float>();
    return std::vector<float>(output_data, output_data + output_size);
  }

  void CompareOutputs(const std::vector<float>& cpu_output,
                      const std::vector<float>& gpu_output, float tolerance = 1e-4f) {
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

// Reference MatMul implementation for testing
void ReferenceMatMul(const float* a, const float* b, float* y, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      y[i * n + j] = sum;
    }
  }
}

// Reference Gemm implementation for testing
void ReferenceGemm(const float* a, const float* b, const float* c, float* y, int m, int k,
                   int n, bool trans_a, bool trans_b, float alpha, float beta) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        float a_val = trans_a ? a[l * m + i] : a[i * k + l];
        float b_val = trans_b ? b[j * k + l] : b[l * n + j];
        sum += a_val * b_val;
      }
      float c_val = (c != nullptr) ? c[i * n + j] : 0.0f;
      y[i * n + j] = alpha * sum + beta * c_val;
    }
  }
}

TEST_F(HipDNNMatMulTest, ReferenceMatMulCorrectness) {
  // Test the reference implementation
  const int m = 2, k = 3, n = 2;
  std::vector<float> a = {1, 2, 3, 4, 5, 6};     // 2x3
  std::vector<float> b = {7, 8, 9, 10, 11, 12};  // 3x2
  std::vector<float> y(m * n, 0.0f);

  ReferenceMatMul(a.data(), b.data(), y.data(), m, k, n);

  // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
  //         = [[58, 64], [139, 154]]
  EXPECT_NEAR(y[0], 58.0f, 1e-5f);
  EXPECT_NEAR(y[1], 64.0f, 1e-5f);
  EXPECT_NEAR(y[2], 139.0f, 1e-5f);
  EXPECT_NEAR(y[3], 154.0f, 1e-5f);
}

TEST_F(HipDNNMatMulTest, BasicMatMul) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(MATMUL_TEST_MODEL_PATH))
      << "MatMul test model not available at: " << MATMUL_TEST_MODEL_PATH;

  // Model parameters: A[64,128] @ B[128,32] = Y[64,32]
  const int64_t m = 64, k = 128, n = 32;

  // Create input data
  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);

  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  std::vector<std::vector<float>> inputs = {a_data, b_data};
  std::vector<std::vector<int64_t>> shapes = {{m, k}, {k, n}};
  std::vector<const char*> names = {"A", "B"};

  // Run with CPU EP
  std::cout << "Running MatMul with CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(MATMUL_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  // Run with HipDNN EP
  std::cout << "Running MatMul with HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(MATMUL_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  // Compare outputs
  CompareOutputs(cpu_output, gpu_output);
}

TEST_F(HipDNNMatMulTest, BasicGemm) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(GEMM_TEST_MODEL_PATH))
      << "Gemm test model not available at: " << GEMM_TEST_MODEL_PATH;

  const int64_t m = 64, k = 128, n = 32;

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);

  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  std::vector<std::vector<float>> inputs = {a_data, b_data};
  std::vector<std::vector<int64_t>> shapes = {{m, k}, {k, n}};
  std::vector<const char*> names = {"A", "B"};

  std::cout << "Running Gemm with CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(GEMM_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  std::cout << "Running Gemm with HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(GEMM_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  CompareOutputs(cpu_output, gpu_output);
}

TEST_F(HipDNNMatMulTest, GemmWithBias) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(GEMM_BIAS_TEST_MODEL_PATH))
      << "Gemm bias test model not available at: " << GEMM_BIAS_TEST_MODEL_PATH;

  const int64_t m = 64, k = 128, n = 32;

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);
  std::vector<float> c_data(m * n);

  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }
  for (size_t i = 0; i < c_data.size(); ++i) {
    c_data[i] = static_cast<float>((i + 5) % 10) / 10.0f;
  }

  std::vector<std::vector<float>> inputs = {a_data, b_data, c_data};
  std::vector<std::vector<int64_t>> shapes = {{m, k}, {k, n}, {m, n}};
  std::vector<const char*> names = {"A", "B", "C"};

  std::cout << "Running Gemm with bias using CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(GEMM_BIAS_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  std::cout << "Running Gemm with bias using HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(GEMM_BIAS_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  CompareOutputs(cpu_output, gpu_output);
}

TEST_F(HipDNNMatMulTest, GemmWithTransposeA) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(GEMM_TRANS_A_TEST_MODEL_PATH))
      << "Gemm transA test model not available at: " << GEMM_TRANS_A_TEST_MODEL_PATH;

  const int64_t m = 64, k = 128, n = 32;
  // With transA=1: A shape is [k, m] = [128, 64]
  std::vector<float> a_data(k * m);
  std::vector<float> b_data(k * n);

  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  std::vector<std::vector<float>> inputs = {a_data, b_data};
  std::vector<std::vector<int64_t>> shapes = {{k, m}, {k, n}};
  std::vector<const char*> names = {"A", "B"};

  std::cout << "Running Gemm with transA using CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(GEMM_TRANS_A_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  std::cout << "Running Gemm with transA using HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(GEMM_TRANS_A_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  CompareOutputs(cpu_output, gpu_output);
}

TEST_F(HipDNNMatMulTest, GemmWithTransposeB) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(GEMM_TRANS_B_TEST_MODEL_PATH))
      << "Gemm transB test model not available at: " << GEMM_TRANS_B_TEST_MODEL_PATH;

  const int64_t m = 64, k = 128, n = 32;
  // With transB=1: B shape is [n, k] = [32, 128]
  std::vector<float> a_data(m * k);
  std::vector<float> b_data(n * k);

  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  std::vector<std::vector<float>> inputs = {a_data, b_data};
  std::vector<std::vector<int64_t>> shapes = {{m, k}, {n, k}};
  std::vector<const char*> names = {"A", "B"};

  std::cout << "Running Gemm with transB using CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(GEMM_TRANS_B_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  std::cout << "Running Gemm with transB using HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(GEMM_TRANS_B_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  CompareOutputs(cpu_output, gpu_output);
}

TEST_F(HipDNNMatMulTest, GemmWithScaling) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(IsModelAvailable(GEMM_SCALED_TEST_MODEL_PATH))
      << "Gemm scaled test model not available at: " << GEMM_SCALED_TEST_MODEL_PATH;

  const int64_t m = 64, k = 128, n = 32;

  std::vector<float> a_data(m * k);
  std::vector<float> b_data(k * n);
  std::vector<float> c_data(m * n);

  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }
  for (size_t i = 0; i < c_data.size(); ++i) {
    c_data[i] = static_cast<float>((i + 5) % 10) / 10.0f;
  }

  std::vector<std::vector<float>> inputs = {a_data, b_data, c_data};
  std::vector<std::vector<int64_t>> shapes = {{m, k}, {k, n}, {m, n}};
  std::vector<const char*> names = {"A", "B", "C"};

  std::cout << "Running Gemm with scaling using CPU EP..." << std::endl;
  auto cpu_output = RunWithCpuEp(GEMM_SCALED_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  std::cout << "Running Gemm with scaling using HipDNN EP..." << std::endl;
  auto gpu_output = RunWithHipDNNEp(GEMM_SCALED_TEST_MODEL_PATH, inputs, shapes, names, "Y");

  CompareOutputs(cpu_output, gpu_output);
}
