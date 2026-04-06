// Copyright (c) 2025, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/core/ep.h"
#include "hipdnn_ep/core/ep_factory.h"
#include "hipdnn_ep/core/kernel.h"
#include "hipdnn_ep/core/node_compute_info.h"
#include "hipdnn_ep/utils/ep_utils.h"

#include "hipdnn_ep/blas_graph/blas_graph.h"

#include <algorithm>
#include <hipdnn_backend.h>

namespace hipdnn_ep {

namespace {

// Check if a Conv node is supported by this EP
static bool IsSupportedConv(Ort::ConstNode node) {
  try {
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

    // Conv requires at least 2 inputs (X, W) and optionally bias
    if (inputs.size() < 2 || outputs.size() != 1) {
      return false;
    }

    // Check data types - we support float and float16
    ONNXTensorElementDataType x_type = GetTensorElementType(inputs[0]);
    ONNXTensorElementDataType w_type = GetTensorElementType(inputs[1]);
    ONNXTensorElementDataType y_type = GetTensorElementType(outputs[0]);

    bool supported_type =
        (x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         x_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
        x_type == w_type && x_type == y_type;

    if (!supported_type) {
      return false;
    }

    // Check if it's a 2D convolution (4D tensors: NCHW)
    auto x_shape = GetTensorShape(inputs[0]);
    auto w_shape = GetTensorShape(inputs[1]);

    if (!x_shape.has_value() || !w_shape.has_value()) {
      return false;  // Dynamic shapes not supported yet
    }

    if (x_shape->size() != 4 || w_shape->size() != 4) {
      return false;  // Only 2D conv supported
    }

    // Check auto_pad - only NOTSET supported (explicit padding)
    std::string auto_pad = GetStringAttrOrDefault(node, "auto_pad", "NOTSET");
    if (auto_pad != "NOTSET") {
      return false;
    }

    // Check group - only 1 supported (no grouped/depthwise convolutions)
    int64_t group = GetIntAttrOrDefault(node, "group", 1);
    if (group != 1) {
      return false;
    }

    // Check dilations - only [1,1] supported (no dilated convolutions)
    std::vector<int64_t> dilations = GetIntsAttrOrDefault(node, "dilations", {1, 1});
    if (dilations.size() != 2 || dilations[0] != 1 || dilations[1] != 1) {
      return false;
    }

    // Check bias (3rd input) if present.
    // Supported shapes: [C_out] (per-channel) or scalar (element count == 1).
    if (inputs.size() >= 3) {
      ONNXTensorElementDataType b_type = GetTensorElementType(inputs[2]);
      if (b_type != x_type) {
        return false;  // Bias type must match input type
      }

      auto b_shape = GetTensorShape(inputs[2]);
      if (!b_shape.has_value()) {
        return false;
      }

      int64_t b_numel = 1;
      for (int64_t d : b_shape.value()) {
        b_numel *= d;
      }

      int64_t c_out = (*w_shape)[0];
      if (b_numel != 1 && !(b_shape->size() == 1 && (*b_shape)[0] == c_out)) {
        return false;  // Bias must be scalar or [C_out]
      }
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if a MatMul node is supported by this EP
static bool IsSupportedMatMul(Ort::ConstNode node) {
  try {
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

    // MatMul requires exactly 2 inputs and 1 output
    if (inputs.size() != 2 || outputs.size() != 1) {
      return false;
    }

    // Check data types - we support float and float16
    ONNXTensorElementDataType a_type = GetTensorElementType(inputs[0]);
    ONNXTensorElementDataType b_type = GetTensorElementType(inputs[1]);
    ONNXTensorElementDataType y_type = GetTensorElementType(outputs[0]);

    bool supported_type =
        (a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
        a_type == b_type && a_type == y_type;

    if (!supported_type) {
      return false;
    }

    // Check shapes - only 2D matrices supported (no batched matmul)
    auto a_shape = GetTensorShape(inputs[0]);
    auto b_shape = GetTensorShape(inputs[1]);

    if (!a_shape.has_value() || !b_shape.has_value()) {
      return false;  // Dynamic shapes not supported yet
    }

    if (a_shape->size() != 2 || b_shape->size() != 2) {
      return false;  // Only 2D matrices supported
    }

    // Verify dimension compatibility: A[M,K] @ B[K,N] = Y[M,N]
    if ((*a_shape)[1] != (*b_shape)[0]) {
      return false;  // K dimensions must match
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if a Gemm node is supported by this EP
static bool IsSupportedGemm(Ort::ConstNode node) {
  try {
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

    // Gemm requires 2-3 inputs (A, B, optional C) and 1 output
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
      return false;
    }

    // Check data types - we support float and float16
    ONNXTensorElementDataType a_type = GetTensorElementType(inputs[0]);
    ONNXTensorElementDataType b_type = GetTensorElementType(inputs[1]);
    ONNXTensorElementDataType y_type = GetTensorElementType(outputs[0]);

    bool supported_type =
        (a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         a_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
        a_type == b_type && a_type == y_type;

    if (!supported_type) {
      return false;
    }

    // Check C input type if present
    if (inputs.size() == 3) {
      ONNXTensorElementDataType c_type = GetTensorElementType(inputs[2]);
      if (c_type != a_type) {
        return false;
      }
    }

    // Check shapes - only 2D matrices supported
    auto a_shape = GetTensorShape(inputs[0]);
    auto b_shape = GetTensorShape(inputs[1]);

    if (!a_shape.has_value() || !b_shape.has_value()) {
      return false;  // Dynamic shapes not supported yet
    }

    if (a_shape->size() != 2 || b_shape->size() != 2) {
      return false;  // Only 2D matrices supported
    }

    // Get transpose attributes
    int64_t trans_a = GetIntAttrOrDefault(node, "transA", 0);
    int64_t trans_b = GetIntAttrOrDefault(node, "transB", 0);

    // Compute effective dimensions after transpose
    // A: transA ? (K, M) : (M, K)
    // B: transB ? (N, K) : (K, N)
    int64_t a_k = trans_a ? (*a_shape)[0] : (*a_shape)[1];
    int64_t b_k = trans_b ? (*b_shape)[1] : (*b_shape)[0];

    if (a_k != b_k) {
      return false;  // K dimensions must match
    }

    // Check C shape if present.
    // Supported shapes:
    //   - scalar (element count == 1)
    //   - 1-D [N] (broadcast along M dimension)
    //   - 2-D [M, N] (exact match)
    // Scalar constant initializers are embedded at graph-build time; runtime
    // scalar inputs become [1]-shaped tensors that hipDNN broadcasts via
    // pointwise ADD.
    if (inputs.size() == 3) {
      auto c_shape = GetTensorShape(inputs[2]);
      if (!c_shape.has_value()) {
        return false;
      }

      int64_t c_numel = 1;
      for (int64_t d : c_shape.value()) {
        c_numel *= d;
      }

      // Scalar bias is always supported.
      if (c_numel == 1) {
        return true;
      }

      int64_t n = trans_b ? (*b_shape)[0] : (*b_shape)[1];

      // 1-D bias [N] — broadcasts along the M dimension.
      if (c_shape->size() == 1 && (*c_shape)[0] == n) {
        return true;
      }

      // 2-D bias must match [M, N] exactly.
      if (c_shape->size() != 2) {
        return false;
      }
      int64_t m = trans_a ? (*a_shape)[1] : (*a_shape)[0];
      if ((*c_shape)[0] != m || (*c_shape)[1] != n) {
        return false;
      }
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if a MultiHeadAttention contrib op is supported by this EP.
// For the initial implementation we support the basic case:
//   - Q, K, V all present as separate 3D tensors [B, S, hidden_size]
//   - No past_key/past_value (KV cache)
//   - No key_padding_mask or attention_bias
//   - Optional scale and causal masking (unidirectional)
static bool IsSupportedMultiHeadAttention(Ort::ConstNode node) {
  try {
    // Must be in the Microsoft contrib domain.
    std::string domain = node.GetDomain();
    if (domain != "com.microsoft") {
      return false;
    }

    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

    // MHA has up to 10 inputs; first 3 (query, key, value) are required
    // for our supported configuration.
    if (inputs.size() < 3) {
      return false;
    }

    // Only single output (no present_key/present_value/qk).
    if (outputs.size() != 1) {
      return false;
    }

    // Check Q, K, V data types — must all match and be float or float16.
    ONNXTensorElementDataType q_type = GetTensorElementType(inputs[0]);
    ONNXTensorElementDataType k_type = GetTensorElementType(inputs[1]);
    ONNXTensorElementDataType v_type = GetTensorElementType(inputs[2]);
    ONNXTensorElementDataType o_type = GetTensorElementType(outputs[0]);

    bool supported_type =
        (q_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         q_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
        q_type == k_type && q_type == v_type && q_type == o_type;

    if (!supported_type) {
      return false;
    }

    // Q, K, V must be 3D: [batch_size, seq_len, hidden_size]
    auto q_shape = GetTensorShape(inputs[0]);
    auto k_shape = GetTensorShape(inputs[1]);
    auto v_shape = GetTensorShape(inputs[2]);

    if (!q_shape.has_value() || !k_shape.has_value() || !v_shape.has_value()) {
      return false;  // Dynamic shapes not supported yet
    }

    if (q_shape->size() != 3 || k_shape->size() != 3 || v_shape->size() != 3) {
      return false;  // Only 3D inputs [B, S, hidden_size] supported
    }

    // num_heads is required.
    int64_t num_heads = GetIntAttrOrDefault(node, "num_heads", 0);
    if (num_heads <= 0) {
      return false;
    }

    // hidden_size must be divisible by num_heads.
    int64_t q_hidden = (*q_shape)[2];
    if (q_hidden % num_heads != 0) {
      return false;
    }

    // Batch dimensions must match.
    if ((*q_shape)[0] != (*k_shape)[0] || (*q_shape)[0] != (*v_shape)[0]) {
      return false;
    }

    // K and V sequence lengths must match.
    if ((*k_shape)[1] != (*v_shape)[1]) {
      return false;
    }

    // K and V hidden sizes must match Q (same head_size * num_heads).
    // hipDNN SDPA supports different num_heads for Q vs KV (GQA), but
    // ORT's MultiHeadAttention uses the same num_heads for all.
    if ((*k_shape)[2] != q_hidden || (*v_shape)[2] != q_hidden) {
      return false;
    }

    // Reject unsupported optional inputs.
    // Input 3 (bias), 4 (key_padding_mask), 6+ (past_key, past_value, etc.)
    // must be absent.
    auto is_present = [&](size_t idx) {
      if (idx >= inputs.size()) return false;
      return GetTensorElementType(inputs[idx]) !=
             ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    };

    // Reject bias (input 3) — would require separate handling.
    if (is_present(3)) {
      return false;
    }

    // Reject key_padding_mask (input 4).
    if (is_present(4)) {
      return false;
    }

    // Reject attention_bias (input 5) — not yet supported.
    if (is_present(5)) {
      return false;
    }

    // Reject past_key (6), past_value (7), past_sequence_length (8),
    // cache_indirection (9).
    for (size_t i = 6; i < inputs.size(); ++i) {
      if (is_present(i)) {
        return false;
      }
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if a GroupQueryAttention contrib op is supported by this EP.
// GQA is like MHA but allows different head counts for Q vs K/V:
//   Q shape: [B, S_q, num_heads * head_size]
//   K shape: [B, S_kv, kv_num_heads * head_size]
//   V shape: [B, S_kv, kv_num_heads * head_size]
// hipDNN's graph.sdpa() natively supports different head counts.
static bool IsSupportedGroupQueryAttention(Ort::ConstNode node) {
  try {
    std::string domain = node.GetDomain();
    if (domain != "com.microsoft") {
      return false;
    }

    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

    // GQA has up to 9 inputs; first 3 (query, key, value) are required.
    if (inputs.size() < 3) {
      return false;
    }

    // Only single output (no present_key/present_value).
    if (outputs.size() != 1) {
      return false;
    }

    // Check Q, K, V data types — must all match and be float or float16.
    ONNXTensorElementDataType q_type = GetTensorElementType(inputs[0]);
    ONNXTensorElementDataType k_type = GetTensorElementType(inputs[1]);
    ONNXTensorElementDataType v_type = GetTensorElementType(inputs[2]);
    ONNXTensorElementDataType o_type = GetTensorElementType(outputs[0]);

    bool supported_type =
        (q_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         q_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
        q_type == k_type && q_type == v_type && q_type == o_type;

    if (!supported_type) {
      return false;
    }

    // Q, K, V must be 3D: [batch_size, seq_len, hidden_size]
    auto q_shape = GetTensorShape(inputs[0]);
    auto k_shape = GetTensorShape(inputs[1]);
    auto v_shape = GetTensorShape(inputs[2]);

    if (!q_shape.has_value() || !k_shape.has_value() || !v_shape.has_value()) {
      return false;
    }

    if (q_shape->size() != 3 || k_shape->size() != 3 || v_shape->size() != 3) {
      return false;
    }

    // num_heads and kv_num_heads are required.
    int64_t num_heads = GetIntAttrOrDefault(node, "num_heads", 0);
    int64_t kv_num_heads = GetIntAttrOrDefault(node, "kv_num_heads", 0);
    if (num_heads <= 0 || kv_num_heads <= 0) {
      return false;
    }

    // num_heads must be divisible by kv_num_heads for even head grouping.
    if (num_heads % kv_num_heads != 0) {
      return false;
    }

    // Q hidden_size must be divisible by num_heads.
    int64_t q_hidden = (*q_shape)[2];
    if (q_hidden % num_heads != 0) {
      return false;
    }

    // K/V hidden_size must be divisible by kv_num_heads.
    int64_t kv_hidden = (*k_shape)[2];
    if (kv_hidden % kv_num_heads != 0) {
      return false;
    }

    // head_size must be consistent: Q and K/V must use the same head_size.
    int64_t head_size = q_hidden / num_heads;
    int64_t kv_head_size = kv_hidden / kv_num_heads;
    if (head_size != kv_head_size) {
      return false;
    }

    // Batch dimensions must match.
    if ((*q_shape)[0] != (*k_shape)[0] || (*q_shape)[0] != (*v_shape)[0]) {
      return false;
    }

    // K and V sequence lengths must match.
    if ((*k_shape)[1] != (*v_shape)[1]) {
      return false;
    }

    // K and V hidden sizes must match each other.
    if ((*v_shape)[2] != kv_hidden) {
      return false;
    }

    // Reject unsupported optional inputs.
    auto is_present = [&](size_t idx) {
      if (idx >= inputs.size()) return false;
      return GetTensorElementType(inputs[idx]) !=
             ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    };

    // Reject past_key (3), past_value (4), seqlens_k (5),
    // total_sequence_length (6), cos_cache (7), sin_cache (8).
    for (size_t i = 3; i < inputs.size(); ++i) {
      if (is_present(i)) {
        return false;
      }
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if a unary pointwise op (Sigmoid) is supported by this EP
static bool IsSupportedUnaryPointwise(Ort::ConstNode node) {
  try {
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

    // Unary pointwise ops require exactly 1 input and 1 output
    if (inputs.size() != 1 || outputs.size() != 1) {
      return false;
    }

    // Check data types - input and output must share the same element type,
    // and must be a supported floating-point type.
    ONNXTensorElementDataType x_type = GetTensorElementType(inputs[0]);
    ONNXTensorElementDataType y_type = GetTensorElementType(outputs[0]);

    if (x_type != y_type) {
      return false;
    }

    if (x_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        x_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      return false;
    }

    // Input must have a static shape
    auto x_shape = GetTensorShape(inputs[0]);
    if (!x_shape.has_value()) {
      return false;  // Dynamic shapes not supported yet
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if a pointwise binary op (Mul, Sub, Add, Div) is supported by this EP
static bool IsSupportedPointwise(Ort::ConstNode node) {
  try {
    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

    // Pointwise binary ops require exactly 2 inputs and 1 output
    if (inputs.size() != 2 || outputs.size() != 1) {
      return false;
    }

    // Check data types - all inputs and output must share the same element type.
    // The hipDNN backend handles type compatibility, so we accept any ONNX type.
    ONNXTensorElementDataType a_type = GetTensorElementType(inputs[0]);
    ONNXTensorElementDataType b_type = GetTensorElementType(inputs[1]);
    ONNXTensorElementDataType y_type = GetTensorElementType(outputs[0]);

    if (a_type != b_type || a_type != y_type) {
      return false;
    }

    // Both inputs must have static shapes
    auto a_shape = GetTensorShape(inputs[0]);
    auto b_shape = GetTensorShape(inputs[1]);

    if (!a_shape.has_value() || !b_shape.has_value()) {
      return false;  // Dynamic shapes not supported yet
    }

    // Check shape compatibility.  We support:
    //   - Exact shape match
    //   - One or both inputs are scalar (rank 0, or every dim is 1)
    // General broadcasting is not supported.
    auto is_scalar = [](const std::vector<int64_t>& shape) {
      return shape.empty() ||
             std::all_of(shape.begin(), shape.end(),
                         [](int64_t d) { return d == 1; });
    };

    if (*a_shape != *b_shape && !is_scalar(*a_shape) && !is_scalar(*b_shape)) {
      return false;
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if a MatMulNBits (com.microsoft) node is supported by this EP.
// MatMulNBits performs Y = A @ dequantize(B) where A is fp16/fp32 and B is
// int4-quantized weights packed as uint8.  We keep the weights in int4 on the
// GPU and use hipDNN's block_scale_dequantize fused with matmul for execution.
// B, scales, and zero_points must be constant initializers.  Only symmetric
// quantization (zero_point = 8, the default) is supported.
static bool IsSupportedMatMulNBits(Ort::ConstNode node) {
  try {
    // MatMulNBits is a com.microsoft contrib op.
    std::string domain = node.GetDomain();
    if (domain != "com.microsoft") {
      return false;
    }

    std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

    // MatMulNBits requires at least 3 inputs (A, B, scales) and 1 output
    if (inputs.size() < 3 || outputs.size() != 1) {
      return false;
    }

    // Check A data type - we support float and float16
    ONNXTensorElementDataType a_type = GetTensorElementType(inputs[0]);
    if (a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        a_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      return false;
    }

    // B must be uint8 (packed int4)
    ONNXTensorElementDataType b_type = GetTensorElementType(inputs[1]);
    if (b_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      return false;
    }

    // scales must match A type
    ONNXTensorElementDataType scales_type = GetTensorElementType(inputs[2]);
    if (scales_type != a_type) {
      return false;
    }

    // Output type must match A type
    ONNXTensorElementDataType y_type = GetTensorElementType(outputs[0]);
    if (y_type != a_type) {
      return false;
    }

    // Check bits attribute - only 4-bit quantization supported
    int64_t bits = GetIntAttrOrDefault(node, "bits", 4);
    if (bits != 4) {
      return false;
    }

    // Check required attributes exist
    int64_t K = GetIntAttrOrDefault(node, "K", 0);
    int64_t N = GetIntAttrOrDefault(node, "N", 0);
    int64_t block_size = GetIntAttrOrDefault(node, "block_size", 0);
    if (K == 0 || N == 0 || block_size == 0) {
      return false;
    }

    // block_size must be even for int4 packing (2 values per byte).
    // The ONNX spec requires power-of-2 >= 16, but validate defensively.
    if (block_size % 2 != 0) {
      return false;
    }

    // A must have static shape and be 2D.  Batched MatMulNBits (3D+ A) is
    // not yet supported because the block_scale_dequantize + matmul graph
    // fusion has only been validated with 2D inputs.
    auto a_shape = GetTensorShape(inputs[0]);
    if (!a_shape.has_value() || a_shape->size() != 2) {
      return false;
    }

    // B, scales must be constant initializers (repacked at build time)
    if (!inputs[1].IsConstantInitializer()) {
      return false;
    }
    if (!inputs[2].IsConstantInitializer()) {
      return false;
    }

    // zero_points (input 3), if present, must also be a constant initializer
    if (inputs.size() >= 4) {
      if (!inputs[3].IsConstantInitializer()) {
        return false;
      }
    }

    return true;

  } catch (...) {
    return false;
  }
}

// Check if an op is supported by this EP
static bool IsSupportedOp(Ort::ConstNode node, bool matmul_supported) {
  std::string op_type = node.GetOperatorType();

  if (op_type == "Conv") {
    return IsSupportedConv(node);
  }

  if (matmul_supported) {
    if (op_type == "MatMul") {
      return IsSupportedMatMul(node);
    }

    if (op_type == "Gemm") {
      return IsSupportedGemm(node);
    }
  }

  // MatMulNBits uses the hipDNN graph path (not hipBLAS-LT), so it does not
  // require hipblaslt_handle_.
  if (op_type == "MatMulNBits") {
    return IsSupportedMatMulNBits(node);
  }

  // Pointwise binary ops.  Keep this list in sync with GetPointwiseMode()
  // in src/hipdnn_graph/hipdnn_graph.cc.
  if (op_type == "Mul" || op_type == "Sub" || op_type == "Add" ||
      op_type == "Div") {
    return IsSupportedPointwise(node);
  }

  // Unary pointwise ops.  Keep this list in sync with GetUnaryPointwiseMode()
  // in src/hipdnn_graph/hipdnn_graph.cc.
  if (op_type == "Sigmoid") {
    return IsSupportedUnaryPointwise(node);
  }

  // Contrib ops (com.microsoft domain)
  if (op_type == "MultiHeadAttention") {
    return IsSupportedMultiHeadAttention(node);
  }
  if (op_type == "GroupQueryAttention") {
    return IsSupportedGroupQueryAttention(node);
  }

  return false;
}

}  // namespace

HipDNNEp::HipDNNEp(HipDNNEpFactory& factory, const Config& config, const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs(static_cast<const ApiPtrs&>(factory)),
      factory_(factory),
      config_(config),
      logger_(logger) {
  // TODO: Do better version management.
  ort_version_supported = ORT_API_VERSION;

  // Initialize function pointers
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  CreateAllocator = CreateAllocatorImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  // Initialize hipDNN
  hipdnnStatus_t status = hipdnnCreate(&hipdnn_handle_);
  if (status != HIPDNN_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create hipDNN handle");
  }

  // Initialize hipBLAS-LT for MatMul/Gemm operations (nullptr if unavailable)
  hipblaslt_handle_ = CreateHipBlasLtHandle();

  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(
      &logger_, ORT_LOGGING_LEVEL_INFO,
      (std::string("HipDNN EP created: ") + factory_.GetName(&factory_)).c_str(),
      EP_FILE, __LINE__, __FUNCTION__));
}

HipDNNEp::~HipDNNEp() {
  kernels_.clear();

  DestroyHipBlasLtHandle(hipblaslt_handle_);
  hipblaslt_handle_ = nullptr;

  if (hipdnn_handle_) {
    hipdnnDestroy(hipdnn_handle_);
    hipdnn_handle_ = nullptr;
  }
}

Kernel* HipDNNEp::GetKernel(const std::string& name) {
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    return nullptr;
  }
  return it->second.get();
}

/*static*/
const char* ORT_API_CALL HipDNNEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const HipDNNEp*>(this_ptr);
  return ep->factory_.GetName(&ep->factory_);
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEp::GetCapabilityImpl(
    OrtEp* this_ptr,
    const OrtGraph* ort_graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    auto* ep = static_cast<HipDNNEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();

    if (nodes.empty()) {
      return nullptr;
    }

    std::vector<Ort::ConstNode> supported_nodes;
    bool matmul_supported = (ep->hipblaslt_handle_ != nullptr);

    for (const auto& node : nodes) {
      if (IsSupportedOp(node, matmul_supported)) {
        supported_nodes.push_back(node);
      }
    }

    if (supported_nodes.empty()) {
      return nullptr;
    }

    LOG(ep->ort_api, ep->logger_, INFO,
        "HipDNN EP: Found " << supported_nodes.size() << " supported nodes");

    // For now, claim nodes individually (no fusion)
    // TODO: Add fusion support for Conv+Bias+Relu patterns
    for (const auto& node : supported_nodes) {
      OrtNodeFusionOptions node_fusion_options = {};
      node_fusion_options.ort_version_supported = ORT_API_VERSION;
      node_fusion_options.drop_constant_initializers = false;  // We need weights

      // ConstNode has implicit conversion to const OrtNode*
      const OrtNode* node_ptr = static_cast<const OrtNode*>(node);
      RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info,
          &node_ptr,
          1,
          &node_fusion_options));
    }

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEp::CompileImpl(
    OrtEp* this_ptr,
    const OrtGraph** ort_graphs,
    const OrtNode** fused_nodes,
    size_t count,
    OrtNodeComputeInfo** node_compute_infos,
    OrtNode** /*ep_context_nodes*/) noexcept {
  try {
    auto* ep = static_cast<HipDNNEp*>(this_ptr);

    for (size_t i = 0; i < count; ++i) {
      Ort::ConstGraph graph{ort_graphs[i]};
      Ort::ConstNode fused_node{fused_nodes[i]};

      std::vector<Ort::ConstNode> nodes = graph.GetNodes();
      if (nodes.empty()) {
        RETURN_ERROR(ep->ort_api, ORT_EP_FAIL, "Empty graph provided for compilation");
      }

      // Create kernel and build/compile the graph
      KernelConfig kernel_config;
      kernel_config.setHipDNNHandle(ep->hipdnn_handle_)
          .setHipBlasLtHandle(ep->hipblaslt_handle_);
      if (ep->config_.use_torch_mlir) {
        kernel_config.setUseTorchMlir()
            .setDumpInputModule(ep->config_.dump_input_module)
            .setDumpLoweredModule(ep->config_.dump_lowered_module);
      }
      auto kernel = std::make_unique<Kernel>(ep->ort_api, ep->logger_, std::move(kernel_config));
      RETURN_IF_ERROR(kernel->BuildAndCompile(graph));

      std::string fused_node_name = fused_node.GetName();
      ep->kernels_.emplace(fused_node_name, std::move(kernel));

      // Create node compute info
      auto compute_info = std::make_unique<NodeComputeInfo>(*ep);
      node_compute_infos[i] = compute_info.release();
    }

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL HipDNNEp::ReleaseNodeComputeInfosImpl(
    OrtEp* /*this_ptr*/,
    OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete static_cast<NodeComputeInfo*>(node_compute_infos[i]);
  }
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEp::CreateAllocatorImpl(
    OrtEp* this_ptr,
    const OrtMemoryInfo* memory_info,
    OrtAllocator** allocator) noexcept {
  auto* ep = static_cast<HipDNNEp*>(this_ptr);
  return ep->factory_.CreateAllocator(&ep->factory_, memory_info, nullptr, allocator);
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEp::CreateSyncStreamForDeviceImpl(
    OrtEp* /*this_ptr*/,
    const OrtMemoryDevice* /*memory_device*/,
    OrtSyncStreamImpl** stream) noexcept {
  // TODO: Implement stream support
  *stream = nullptr;
  return nullptr;
}

}  // namespace hipdnn_ep
