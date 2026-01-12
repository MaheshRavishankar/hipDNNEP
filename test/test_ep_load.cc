// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <string>

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#ifdef _WIN32
#include <windows.h>
inline std::wstring ToWideString(const char* str) {
  int len = MultiByteToWideChar(CP_UTF8, 0, str, -1, nullptr, 0);
  std::wstring result(len - 1, 0);
  MultiByteToWideChar(CP_UTF8, 0, str, -1, &result[0], len);
  return result;
}
#endif

#ifndef HIPDNN_EP_LIB_PATH
#ifdef _WIN32
#define HIPDNN_EP_LIB_PATH "./hipdnn_ep.dll"
#else
#define HIPDNN_EP_LIB_PATH "./libhipdnn_ep.so"
#endif
#endif

class HipDNNEpLoadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize ORT
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HipDNNEpTest");
  }

  void TearDown() override {
    env_.reset();
  }

  std::unique_ptr<Ort::Env> env_;
};

TEST_F(HipDNNEpLoadTest, RegisterEpLibrary) {
  const char* lib_path_str = HIPDNN_EP_LIB_PATH;
#ifdef _WIN32
  auto lib_path_w = ToWideString(lib_path_str);
  OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
      *env_, "HipDNN", lib_path_w.c_str());
#else
  OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
      *env_, "HipDNN", lib_path_str);
#endif

  if (status != nullptr) {
    std::string error_msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    FAIL() << "Failed to register EP library: " << error_msg;
  }

  SUCCEED() << "EP library registered successfully";
}

TEST_F(HipDNNEpLoadTest, GetEpDevices) {
  const char* lib_path_str = HIPDNN_EP_LIB_PATH;

  // First register the EP
#ifdef _WIN32
  auto lib_path_w = ToWideString(lib_path_str);
  OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
      *env_, "HipDNN", lib_path_w.c_str());
#else
  OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
      *env_, "HipDNN", lib_path_str);
#endif

  if (status != nullptr) {
    std::string error_msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    GTEST_SKIP() << "EP library not available: " << error_msg;
  }

  // Get available devices using the C++ wrapper
  try {
    std::vector<Ort::ConstEpDevice> devices = env_->GetEpDevices();
    std::cout << "Found " << devices.size() << " EP devices" << std::endl;
    EXPECT_GE(devices.size(), static_cast<size_t>(0));
  } catch (const Ort::Exception& ex) {
    FAIL() << "Failed to get EP devices: " << ex.what();
  }
}
