// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.
//
// Central header for ONNX Runtime C++ API includes.
//
// This header ensures ORT_API_MANUAL_INIT is defined before including
// the ORT C++ API. This is required for Execution Provider plugins
// that are dynamically loaded - the ORT API is already initialized
// by the host application, so the plugin must not reinitialize it.

#pragma once

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif

#include "onnxruntime_cxx_api.h"

#include <sstream>

//===----------------------------------------------------------------------===//
// Error handling macros
//===----------------------------------------------------------------------===//

#define RETURN_IF_ERROR(fn)     \
  do {                          \
    Ort::Status _status{(fn)};  \
    if (!_status.IsOK()) {      \
      return _status.release(); \
    }                           \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

#define HIPDNN_EP_ENFORCE(condition, ...)                       \
  do {                                                          \
    if (!(condition)) {                                         \
      std::ostringstream oss;                                   \
      oss << "HIPDNN_EP_ENFORCE failed: " << #condition << " "; \
      oss << __VA_ARGS__;                                       \
      throw std::runtime_error(oss.str());                      \
    }                                                           \
  } while (false)

#define IGNORE_ORTSTATUS(status_expr)   \
  do {                                  \
    OrtStatus* _status = (status_expr); \
    Ort::Status _ignored{_status};      \
  } while (false)

#ifdef _WIN32
#define EP_WSTR(x) L##x
#define EP_FILE_INTERNAL(x) EP_WSTR(x)
#define EP_FILE EP_FILE_INTERNAL(__FILE__)
#else
#define EP_FILE __FILE__
#endif

namespace hipdnn_ep {

/// API pointers structure - holds references to ORT API interfaces.
struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

}  // namespace hipdnn_ep

//===----------------------------------------------------------------------===//
// Logging macros
//===----------------------------------------------------------------------===//

#define LOG(api, logger, level, ...)                                                              \
  do {                                                                                            \
    std::ostringstream ss;                                                                        \
    ss << __VA_ARGS__;                                                                            \
    IGNORE_ORTSTATUS((api).Logger_LogMessage(&(logger), ORT_LOGGING_LEVEL_##level,                \
                                             ss.str().c_str(), EP_FILE, __LINE__, __FUNCTION__)); \
  } while (false)

#define RETURN_ERROR(api, code, ...)                   \
  do {                                                 \
    std::ostringstream ss;                             \
    ss << __VA_ARGS__;                                 \
    return (api).CreateStatus(code, ss.str().c_str()); \
  } while (false)

namespace hipdnn_ep {

/// Returns an entry in the session option configurations, or a default value if not present.
inline OrtStatus* GetSessionConfigEntryOrDefault(const OrtSessionOptions& session_options,
                                                 const char* config_key, const std::string& default_val,
                                                 /*out*/ std::string& config_val) {
  try {
    Ort::ConstSessionOptions sess_opt{&session_options};
    config_val = sess_opt.GetConfigEntryOrDefault(config_key, default_val);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  }

  return nullptr;
}

}  // namespace hipdnn_ep
