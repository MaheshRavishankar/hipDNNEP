// Copyright (c) 2025, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/utils/ep_utils.h"

namespace hipdnn_ep {

void IsFloatTensor(Ort::ConstValueInfo value_info, bool& result) {
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

std::optional<std::vector<int64_t>> GetTensorShape(Ort::ConstValueInfo value_info) {
  const auto type_info = value_info.TypeInfo();
  const auto onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return std::nullopt;
  }

  const auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetShape();
}

ONNXTensorElementDataType GetTensorElementType(Ort::ConstValueInfo value_info) {
  const auto type_info = value_info.TypeInfo();
  const auto onnx_type = type_info.GetONNXType();
  if (onnx_type != ONNX_TYPE_TENSOR) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  const auto type_shape = type_info.GetTensorTypeAndShapeInfo();
  return type_shape.GetElementType();
}

std::string GetStringAttrOrDefault(Ort::ConstNode node, const char* name, const std::string& default_val) {
  Ort::ConstOpAttr attr{nullptr};
  auto status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || !static_cast<const OrtOpAttr*>(attr)) {
    return default_val;
  }
  std::string value;
  if (!attr.GetValue(value).IsOK()) {
    return default_val;
  }
  return value;
}

int64_t GetIntAttrOrDefault(Ort::ConstNode node, const char* name, int64_t default_val) {
  Ort::ConstOpAttr attr{nullptr};
  auto status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || !static_cast<const OrtOpAttr*>(attr)) {
    return default_val;
  }
  int64_t value;
  if (!attr.GetValue(value).IsOK()) {
    return default_val;
  }
  return value;
}

float GetFloatAttrOrDefault(Ort::ConstNode node, const char* name, float default_val) {
  Ort::ConstOpAttr attr{nullptr};
  auto status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || !static_cast<const OrtOpAttr*>(attr)) {
    return default_val;
  }
  float value;
  if (!attr.GetValue(value).IsOK()) {
    return default_val;
  }
  return value;
}

std::vector<int64_t> GetIntsAttrOrDefault(Ort::ConstNode node, const char* name,
                                          const std::vector<int64_t>& default_val) {
  Ort::ConstOpAttr attr{nullptr};
  auto status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || !static_cast<const OrtOpAttr*>(attr)) {
    return default_val;
  }
  std::vector<int64_t> value;
  if (!attr.GetValueArray(value).IsOK()) {
    return default_val;
  }
  return value;
}

}  // namespace hipdnn_ep
