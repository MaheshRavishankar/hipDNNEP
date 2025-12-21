// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// ORT_API_MANUAL_INIT is typically defined in CMakeLists.txt
// If not, define it here to ensure the API is correctly initialized
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

// Error handling macros
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

// API pointers structure - holds references to ORT API interfaces
struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

// Logging macro (requires 'api_' and 'logger_' in scope)
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

// Helper to check if a value is a float tensor
inline void IsFloatTensor(Ort::ConstValueInfo value_info, bool& result) {
  result = false;

  auto type_info = value_info.TypeInfo();
  ONNXType onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return;
  }

  auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = type_shape.GetElementType();
  if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return;
  }
  result = true;
}

// Gets the tensor shape from `value_info`. Returns std::nullopt if not a tensor.
inline std::optional<std::vector<int64_t>> GetTensorShape(Ort::ConstValueInfo value_info) {
  const auto type_info = value_info.TypeInfo();
  const auto onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return std::nullopt;
  }

  const auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetShape();
}

// Gets the tensor element type. Returns UNDEFINED if not a tensor.
inline ONNXTensorElementDataType GetTensorElementType(Ort::ConstValueInfo value_info) {
  const auto type_info = value_info.TypeInfo();
  const auto onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  const auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetElementType();
}

// Returns an entry in the session option configurations, or a default value if not present.
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

// Helper to get a string attribute with a default value
std::string GetStringAttrOrDefault(Ort::ConstNode node, const char* name, const std::string& default_val);

// Helper to get an int64 attribute with a default value
int64_t GetIntAttrOrDefault(Ort::ConstNode node, const char* name, int64_t default_val);

// Helper to get an int64 array attribute with a default value
std::vector<int64_t> GetIntsAttrOrDefault(Ort::ConstNode node, const char* name,
                                          const std::vector<int64_t>& default_val);

}  // namespace hipdnn_ep
