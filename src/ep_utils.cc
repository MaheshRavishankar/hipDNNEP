// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/ep_utils.h"

namespace hipdnn_ep {

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
