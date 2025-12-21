// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/ep_factory.h"

// Symbol visibility for exported functions
#ifdef _WIN32
#ifdef HIPDNN_EP_EXPORTS
#define HIPDNN_EP_API __declspec(dllexport)
#else
#define HIPDNN_EP_API __declspec(dllimport)
#endif
#else
#ifdef HIPDNN_EP_EXPORTS
#define HIPDNN_EP_API __attribute__((visibility("default")))
#else
#define HIPDNN_EP_API
#endif
#endif

using namespace hipdnn_ep;

extern "C" {

/// @brief Create EP factories - main entry point for plugin EP
/// @param registration_name Name used to register the EP
/// @param ort_api_base Base ORT API for accessing versioned APIs
/// @param default_logger Logger for the EP factory
/// @param factories Output array of factories
/// @param max_factories Maximum number of factories to create
/// @param num_factories Output: number of factories created
/// @return OrtStatus* nullptr on success, error status otherwise
HIPDNN_EP_API OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // Initialize C++ API
  Ort::InitApi(ort_api);

  if (max_factories < 1) {
    *num_factories = 0;
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space for EP factory. Need at least 1.");
  }

  try {
    std::unique_ptr<OrtEpFactory> factory = std::make_unique<HipDNNEpFactory>(
        registration_name,
        ApiPtrs{*ort_api, *ep_api, *model_editor_api},
        *default_logger);

    factories[0] = factory.release();
    *num_factories = 1;

  } catch (const std::exception& ex) {
    return ort_api->CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/// @brief Release an EP factory
/// @param factory Factory to release
/// @return OrtStatus* nullptr on success
HIPDNN_EP_API OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<HipDNNEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
