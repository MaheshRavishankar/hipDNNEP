// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <numeric>

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#ifndef HIPDNN_EP_LIB_PATH
#define HIPDNN_EP_LIB_PATH "./libhipdnn_ep.so"
#endif

#ifndef CONV_TEST_MODEL_PATH
#define CONV_TEST_MODEL_PATH "./conv_test.onnx"
#endif

class HipDNNConvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HipDNNConvTest");

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

    // Check if model file exists
    std::ifstream model_file(CONV_TEST_MODEL_PATH);
    model_available_ = model_file.good();
    if (!model_available_) {
      std::cout << "Model not available at: " << CONV_TEST_MODEL_PATH << std::endl;
    }
  }

  void TearDown() override {
    env_.reset();
  }

  std::unique_ptr<Ort::Env> env_;
  bool ep_available_{false};
  bool model_available_{false};
};

// Simple reference Conv2D implementation for verification
void ReferenceConv2D(
    const float* input, const float* weight, float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w) {

  int H_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
  int W_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

  for (int n = 0; n < N; ++n) {
    for (int c_out = 0; c_out < C_out; ++c_out) {
      for (int h_out = 0; h_out < H_out; ++h_out) {
        for (int w_out = 0; w_out < W_out; ++w_out) {
          float sum = 0.0f;

          for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int k_h = 0; k_h < K_h; ++k_h) {
              for (int k_w = 0; k_w < K_w; ++k_w) {
                int h_in = h_out * stride_h - pad_h + k_h;
                int w_in = w_out * stride_w - pad_w + k_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                  int input_idx = n * C_in * H_in * W_in + c_in * H_in * W_in + h_in * W_in + w_in;
                  int weight_idx = c_out * C_in * K_h * K_w + c_in * K_h * K_w + k_h * K_w + k_w;
                  sum += input[input_idx] * weight[weight_idx];
                }
              }
            }
          }

          int output_idx = n * C_out * H_out * W_out + c_out * H_out * W_out + h_out * W_out + w_out;
          output[output_idx] = sum;
        }
      }
    }
  }
}

TEST_F(HipDNNConvTest, BasicConv2D) {
  ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
  ASSERT_TRUE(model_available_) << "Conv test model not available at: " << CONV_TEST_MODEL_PATH;

  // Model parameters (must match gen_conv_model.py defaults)
  const int64_t N = 1, C = 1, H = 8, W = 8;
  const std::vector<int64_t> input_shape = {N, C, H, W};
  const size_t input_size = N * C * H * W;

  // Create input data
  std::vector<float> input_data(input_size);
  for (size_t i = 0; i < input_size; ++i) {
    input_data[i] = static_cast<float>(i % 10) / 10.0f;
  }

  // Run with CPU EP first to get reference output
  std::vector<float> cpu_output;
  {
    Ort::SessionOptions session_options;
    Ort::Session session(*env_, CONV_TEST_MODEL_PATH, session_options);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_size, input_shape.data(), input_shape.size());

    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};

    auto output_tensors = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

    ASSERT_EQ(output_tensors.size(), 1);
    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_size = output_info.GetElementCount();

    const float* output_data = output_tensor.GetTensorData<float>();
    cpu_output.assign(output_data, output_data + output_size);

    std::cout << "CPU output size: " << output_size << std::endl;
  }

  // Run with HipDNN EP
  std::vector<float> gpu_output;
  {
    // Get EP devices
    std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
    ASSERT_FALSE(devices.empty()) << "No EP devices found";

    // Find a HipDNN device
    const OrtEpDevice* hipdnn_device = nullptr;
    for (const auto& device : devices) {
      std::string ep_name = device.EpName();
      std::cout << "Found EP device: " << ep_name << std::endl;
      if (ep_name == "HipDNN") {
        hipdnn_device = static_cast<const OrtEpDevice*>(device);
        break;
      }
    }

    ASSERT_NE(hipdnn_device, nullptr) << "No HipDNN device found";

    Ort::SessionOptions session_options;

    // Add HipDNN EP using V2 API
    OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        session_options, *env_, &hipdnn_device, 1, nullptr, nullptr, 0);

    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      FAIL() << "Failed to add HipDNN EP: " << error_msg;
    }

    std::cout << "Creating session with HipDNN EP..." << std::endl;
    Ort::Session session(*env_, CONV_TEST_MODEL_PATH, session_options);
    std::cout << "Session created successfully" << std::endl;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_size, input_shape.data(), input_shape.size());

    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};

    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);
    std::cout << "Inference completed" << std::endl;

    ASSERT_EQ(output_tensors.size(), 1);
    auto& output_tensor = output_tensors[0];
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    size_t output_size = output_info.GetElementCount();

    const float* output_data = output_tensor.GetTensorData<float>();
    gpu_output.assign(output_data, output_data + output_size);

    std::cout << "GPU output size: " << output_size << std::endl;
  }

  // Compare outputs
  ASSERT_EQ(cpu_output.size(), gpu_output.size()) << "Output size mismatch";

  float max_diff = 0.0f;
  for (size_t i = 0; i < cpu_output.size(); ++i) {
    float diff = std::abs(cpu_output[i] - gpu_output[i]);
    max_diff = std::max(max_diff, diff);
    EXPECT_NEAR(cpu_output[i], gpu_output[i], 1e-4f)
        << "Mismatch at index " << i << ": CPU=" << cpu_output[i] << ", GPU=" << gpu_output[i];
  }

  std::cout << "Max difference between CPU and GPU: " << max_diff << std::endl;
}

TEST_F(HipDNNConvTest, ReferenceConvCorrectness) {
  // Test the reference implementation
  const int N = 1, C_in = 1, H_in = 4, W_in = 4;
  const int C_out = 1, K_h = 3, K_w = 3;
  const int pad_h = 0, pad_w = 0;
  const int stride_h = 1, stride_w = 1;

  // Simple input: 4x4 matrix of ones
  std::vector<float> input(N * C_in * H_in * W_in, 1.0f);

  // Simple weight: 3x3 matrix of ones
  std::vector<float> weight(C_out * C_in * K_h * K_w, 1.0f);

  // Output should be 2x2 (4 - 3 + 1 = 2)
  int H_out = (H_in - K_h) / stride_h + 1;
  int W_out = (W_in - K_w) / stride_w + 1;
  std::vector<float> output(N * C_out * H_out * W_out, 0.0f);

  ReferenceConv2D(input.data(), weight.data(), output.data(),
                  N, C_in, H_in, W_in, C_out, K_h, K_w,
                  pad_h, pad_w, stride_h, stride_w);

  // Each output should be sum of 3x3 = 9 ones = 9.0
  for (int i = 0; i < static_cast<int>(output.size()); ++i) {
    EXPECT_NEAR(output[i], 9.0f, 1e-5f) << "Output mismatch at index " << i;
  }
}
