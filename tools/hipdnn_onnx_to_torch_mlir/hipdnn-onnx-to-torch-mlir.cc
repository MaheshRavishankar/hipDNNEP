// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.
//
// Tool to load an ONNX model and print the generated Torch-MLIR to stdout.
// Used for lit testing the IR generation.

#include <iostream>
#include <string>

#include "hipdnn_ep/core/ort_api.h"

#ifndef HIPDNN_EP_LIB_PATH
#define HIPDNN_EP_LIB_PATH "./libhipdnn_ep.so"
#endif

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
    return 1;
  }

  const char* model_path = argv[1];

  const char* lib_path = HIPDNN_EP_LIB_PATH;

  try {
    // Initialize ORT
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "HipDNNOnnxToTorchMLIR");

    // Register EP
    OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
        env, "HipDNN", lib_path);
    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      std::cerr << "Failed to register EP library: " << error_msg << "\n";
      return 1;
    }

    // Get EP devices
    std::vector<Ort::ConstEpDevice> devices = env.GetEpDevices();
    const OrtEpDevice* hipdnn_device = nullptr;
    for (const auto& device : devices) {
      if (std::string(device.EpName()) == "HipDNN") {
        hipdnn_device = static_cast<const OrtEpDevice*>(device);
        break;
      }
    }

    if (hipdnn_device == nullptr) {
      std::cerr << "No HipDNN device found\n";
      return 1;
    }

    // Create session options with torch-mlir config
    Ort::SessionOptions session_options;

    // Enable torch-mlir path and dump to stdout
    session_options.AddConfigEntry("hipdnn.use_torch_mlir", "1");
    session_options.AddConfigEntry("hipdnn.dump_torch_mlir", "1");

    // Add HipDNN EP
    status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        session_options, env, &hipdnn_device, 1, nullptr, nullptr, 0);
    if (status != nullptr) {
      std::string error_msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      std::cerr << "Failed to add HipDNN EP: " << error_msg << "\n";
      return 1;
    }

    // Create session - this triggers compilation which prints MLIR to stdout
    Ort::Session session(env, model_path, session_options);

  } catch (const Ort::Exception& ex) {
    std::cerr << "ORT Exception: " << ex.what() << "\n";
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "Exception: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
