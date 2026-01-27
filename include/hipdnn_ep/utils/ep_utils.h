// Copyright (c) 2025, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.
//
// Utility functions for working with ORT graph elements (nodes, values, etc.)

#pragma once

#include "hipdnn_ep/core/ort_api.h"

#include <optional>
#include <string>
#include <vector>

namespace hipdnn_ep {

/// Check if a value is a float tensor.
void IsFloatTensor(Ort::ConstValueInfo value_info, bool& result);

/// Get the tensor shape from value_info. Returns std::nullopt if not a tensor.
std::optional<std::vector<int64_t>> GetTensorShape(Ort::ConstValueInfo value_info);

/// Get the tensor element type. Returns UNDEFINED if not a tensor.
ONNXTensorElementDataType GetTensorElementType(Ort::ConstValueInfo value_info);

/// Get a string attribute with a default value.
std::string GetStringAttrOrDefault(Ort::ConstNode node, const char* name,
                                   const std::string& default_val);

/// Get an int64 attribute with a default value.
int64_t GetIntAttrOrDefault(Ort::ConstNode node, const char* name, int64_t default_val);

/// Get a float attribute with a default value.
float GetFloatAttrOrDefault(Ort::ConstNode node, const char* name, float default_val);

/// Get an int64 array attribute with a default value.
std::vector<int64_t> GetIntsAttrOrDefault(Ort::ConstNode node, const char* name,
                                          const std::vector<int64_t>& default_val);

}  // namespace hipdnn_ep
