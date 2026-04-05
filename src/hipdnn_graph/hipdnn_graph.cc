// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/hipdnn_graph/hipdnn_graph.h"
#include "hipdnn_ep/utils/ep_utils.h"

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>

#include <cassert>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>

#ifdef HIPDNN_EP_HAS_TORCH_MLIR
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#endif

namespace hipdnn_ep {

namespace {

// Layout for 4D convolution tensors.
// NCHW: row-major / channels-first (default)
// NHWC: channels-last
enum class ConvLayout { NCHW,
                        NHWC };

// Compute row-major strides from shape (NCHW / default layout).
static std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// Compute NHWC (channels-last) strides for a 4D shape labeled [N, C, H, W].
// hipDNN always uses the NCHW dimension labeling, so a tensor with physical
// NHWC layout is expressed via strides that make the C dimension the
// fastest-varying and N the slowest.
//
// For a [N, C, H, W] tensor stored in NHWC order in memory the strides are:
//   N stride = H * W * C
//   C stride = 1              (fastest-varying)
//   H stride = W * C
//   W stride = C
static std::vector<int64_t> ComputeNHWCStrides(const std::vector<int64_t>& shape) {
  assert(shape.size() == 4 && "ComputeNHWCStrides requires a 4D shape");
  int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
  (void)N;
  return {H * W * C, 1, W * C, C};
}

// Compute strides for a 4D shape according to the given layout.
static std::vector<int64_t> ComputeConvStrides(
    const std::vector<int64_t>& shape, ConvLayout layout) {
  if (layout == ConvLayout::NHWC) {
    return ComputeNHWCStrides(shape);
  }
  return ComputeStrides(shape);
}

// Convert ONNX tensor element data type to hipDNN data type
std::optional<hipdnn_frontend::DataType> ToHipDNNDataType(ONNXTensorElementDataType onnx_dtype) {
  using hipdnn_frontend::DataType;
  switch (onnx_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::HALF;
    default:
      return std::nullopt;
  }
}

// Check if a hipDNN data type is a supported floating-point type.
static bool IsFloatDataType(hipdnn_frontend::DataType dtype) {
  return dtype == hipdnn_frontend::DataType::FLOAT ||
         dtype == hipdnn_frontend::DataType::HALF;
}

// Determine compute data type based on input data types
// For float types with precision <= float32, compute in float32
std::optional<hipdnn_frontend::DataType> GetComputeDataType(
    hipdnn_frontend::DataType x_dtype,
    hipdnn_frontend::DataType w_dtype) {
  if (IsFloatDataType(x_dtype) && IsFloatDataType(w_dtype)) {
    // Use float32 for compute when inputs are float types with precision <= float32
    return hipdnn_frontend::DataType::FLOAT;
  }

  return std::nullopt;
}

using TensorAttrPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;

// Return true if the tensor has exactly one element (scalar).
static bool IsScalarAttr(const TensorAttrPtr& attr) {
  int64_t numel = 1;
  for (int64_t d : attr->get_dim()) {
    numel *= d;
  }
  return numel == 1;
}

// Create a pass-by-value scalar TensorAttributes for pointwise ops.
// The value is embedded directly in the graph via set_value(), so no
// runtime device pointer is needed at execute time.
static TensorAttrPtr CreateScalarTensorAttr(int64_t uid, float value) {
  auto attr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
  attr->set_value(value);
  attr->set_uid(uid).set_name("constant_" + std::to_string(uid));
  return attr;
}

// Check if a value is a scalar constant initializer and, if so, create a
// pass-by-value TensorAttributes with the original data type preserved.
// A "scalar" here means element count == 1 (shape [] or [1]).
// Returns a valid TensorAttrPtr on success, nullptr otherwise.
//
// The value is stored via TensorAttributes::set_value<T>() using the closest
// type that the hipDNN API supports:
//   ONNX float       -> set_value<float>
//   ONNX float16     -> set_value<half>        (from raw bits)
//   ONNX double      -> set_value<double>
//   ONNX int32       -> set_value<int32_t>
//   ONNX uint8       -> set_value<uint8_t>
//   ONNX int64/int8/int16/uint16 -> set_value<int32_t> (narrowing; the API
//       does not support these types directly)
static TensorAttrPtr TryExtractScalarConstant(Ort::ConstValueInfo value_info) {
  using hipdnn_data_sdk::types::half;
  using hipdnn_frontend::graph::TensorAttributes;

  if (!value_info.IsConstantInitializer()) {
    return nullptr;
  }

  auto shape = GetTensorShape(value_info);
  if (!shape.has_value()) {
    return nullptr;
  }

  // Compute element count — must be exactly 1.
  int64_t numel = 1;
  for (int64_t d : shape.value()) {
    numel *= d;
  }
  if (numel != 1) {
    return nullptr;
  }

  Ort::ConstValue init_value{nullptr};
  Ort::Status status = value_info.GetInitializer(init_value);
  if (!status.IsOK() || init_value == nullptr) {
    return nullptr;
  }

  // Extract the scalar and call set_value() with the original (or closest
  // supported) type so that hipDNN sees the value in its native precision.
  auto attr = std::make_shared<TensorAttributes>();
  ONNXTensorElementDataType elem_type = GetTensorElementType(value_info);
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const float* data = init_value.GetTensorData<float>();
      if (data == nullptr) return nullptr;
      attr->set_value(data[0]);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      const uint16_t* data = init_value.GetTensorData<uint16_t>();
      if (data == nullptr) return nullptr;
      attr->set_value(half::from_bits(data[0]));
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      const double* data = init_value.GetTensorData<double>();
      if (data == nullptr) return nullptr;
      attr->set_value(data[0]);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      const int32_t* data = init_value.GetTensorData<int32_t>();
      if (data == nullptr) return nullptr;
      attr->set_value(data[0]);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      const uint8_t* data = init_value.GetTensorData<uint8_t>();
      if (data == nullptr) return nullptr;
      attr->set_value(data[0]);
      break;
    }
    // Types not directly supported by set_value() — convert to int32_t.
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      const int64_t* data = init_value.GetTensorData<int64_t>();
      if (data == nullptr) return nullptr;
      attr->set_value(static_cast<int32_t>(data[0]));
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      const int8_t* data = init_value.GetTensorData<int8_t>();
      if (data == nullptr) return nullptr;
      attr->set_value(static_cast<int32_t>(data[0]));
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
      const int16_t* data = init_value.GetTensorData<int16_t>();
      if (data == nullptr) return nullptr;
      attr->set_value(static_cast<int32_t>(data[0]));
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      const uint16_t* data = init_value.GetTensorData<uint16_t>();
      if (data == nullptr) return nullptr;
      attr->set_value(static_cast<int32_t>(data[0]));
      break;
    }
    default:
      return nullptr;
  }
  return attr;
}

// Create TensorAttributes from a ConstValueInfo.
// Strides are always computed as row-major (NCHW for 4D tensors).  The
// appropriate strides for a different physical layout (e.g. NHWC) are set
// later by the per-op node builder when the layout is known.
Status CreateTensorAttr(
    Ort::ConstValueInfo value_info,
    int64_t uid,
    TensorAttrPtr& out_attr) {
  using hipdnn_frontend::graph::TensorAttributes;

  std::string name = value_info.GetName();

  auto shape = GetTensorShape(value_info);
  if (!shape.has_value()) {
    return Status::Failure("Value must have static shape: " + name);
  }

  auto dtype = ToHipDNNDataType(GetTensorElementType(value_info));
  if (!dtype.has_value()) {
    return Status::Failure("Unsupported data type for value: " + name);
  }

  out_attr = std::make_shared<TensorAttributes>();
  out_attr->set_uid(uid)
      .set_name(name)
      .set_data_type(dtype.value())
      .set_dim(shape.value())
      .set_stride(ComputeStrides(shape.value()));

  return Status::Success();
}

// Reshape a 1D bias [C] to a 4D broadcast shape compatible with the given
// layout so that hipDNN pointwise ADD can broadcast it over the conv output.
//
//   NCHW -> [1, C, 1, 1]   (C along axis 1)
//   NHWC -> [1, C, 1, 1]   with NHWC strides [C, 1, C, C]
//
// Accepted inputs:
//   - pass-by-value scalar   -> left unchanged (broadcasts naturally)
//   - 1D [C]                 -> reshaped according to layout
//   - 4D (already broadcast) -> left unchanged
static Status ReshapeBiasForConv(const TensorAttrPtr& bias,
                                 ConvLayout layout) {
  // Pass-by-value scalars have dim={1} set by set_value(); leave them alone.
  if (bias->get_pass_by_value()) {
    return Status::Success();
  }

  auto bias_dim = bias->get_dim();

  if (bias_dim.size() == 4) {
    // Already 4D — assume the caller shaped it correctly.
    return Status::Success();
  }

  if (bias_dim.size() == 1) {
    int64_t C = bias_dim[0];
    // hipDNN labels dimensions as [N, C, H, W].  We set dim to [1, C, 1, 1]
    // for both layouts; the strides determine the physical layout.
    bias->set_dim({1, C, 1, 1});
    bias->set_stride(ComputeConvStrides({1, C, 1, 1}, layout));
    return Status::Success();
  }

  return Status::Failure(
      "Conv bias has unsupported rank " + std::to_string(bias_dim.size()) +
      "; expected 1D [C] or 4D [1,C,1,1]");
}

// Read the convolution layout from the ORT node's attributes and domain.
//
// ORT signals NHWC layout through two mechanisms:
//   1. The NhwcTransformer sets a "channels_last" integer attribute (== 1).
//   2. The EP v2 layout transformer moves the node to the
//      "com.ms.internal.nhwc" domain.
//
// Returns NHWC when either signal is present, NCHW otherwise.
static ConvLayout GetConvLayoutFromNode(Ort::ConstNode node) {
  int64_t channels_last = GetIntAttrOrDefault(node, "channels_last", 0);
  if (channels_last == 1) {
    return ConvLayout::NHWC;
  }
  std::string domain = node.GetDomain();
  if (domain == "com.ms.internal.nhwc") {
    return ConvLayout::NHWC;
  }
  return ConvLayout::NCHW;
}

// Add Conv operation to hipDNN graph
// Takes input tensor attributes (X, W, optional B), returns output tensor
// attribute (Y).  B may be a scalar (embedded constant) or a 1D per-channel
// tensor; both are handled via a pointwise ADD that broadcasts over the
// convolution output.
//
// Layout handling is fully self-contained: the function detects NHWC from
// the ORT node's attributes/domain and adjusts dims, strides, and the
// output tensor accordingly.  Callers do not need to know the layout.
Status AddConvNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr,
    int64_t& next_uid) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::ConvolutionMode;
  using hipdnn_frontend::PointwiseMode;

  if (input_attrs.size() < 2) {
    return Status::Failure("Conv requires at least 2 input tensor attributes");
  }

  const auto& x_attr = input_attrs[0];
  const auto& w_attr = input_attrs[1];

  // X and W must be tensors, not scalars.
  if (IsScalarAttr(x_attr)) {
    return Status::Failure("Conv input X must be a tensor, not a scalar");
  }
  if (IsScalarAttr(w_attr)) {
    return Status::Failure("Conv filter W must be a tensor, not a scalar");
  }

  // Detect layout from the ORT node.
  ConvLayout layout = GetConvLayoutFromNode(node);

  // When the node uses NHWC layout, relabel dims to hipDNN's [N,C,H,W]
  // convention and set NHWC strides.  ORT's layout transformer gives us:
  //   X shape: [N, H, W, C] -> relabel to [N, C, H, W]
  //   W shape: [K, C, kH, kW] (unchanged -- only input[0] is transposed)
  // The strides then encode the NHWC physical layout.
  if (layout == ConvLayout::NHWC) {
    auto x_dim = x_attr->get_dim();
    if (x_dim.size() == 4) {
      // ORT shape [N, H, W, C] -> hipDNN dim [N, C, H, W]
      std::vector<int64_t> nchw_dim = {x_dim[0], x_dim[3], x_dim[1], x_dim[2]};
      x_attr->set_dim(nchw_dim);
      x_attr->set_stride(ComputeNHWCStrides(nchw_dim));
    }
    // ORT's layout transformer does NOT transpose the filter -- it remains in
    // NCHW physical layout [K, C, kH, kW].  Keep NCHW (row-major) strides so
    // hipDNN reads the untransposed data correctly.
  }

  // Extract Conv attributes
  std::vector<int64_t> pads = GetIntsAttrOrDefault(node, "pads", {0, 0, 0, 0});
  std::vector<int64_t> strides = GetIntsAttrOrDefault(node, "strides", {1, 1});
  std::vector<int64_t> dilations = GetIntsAttrOrDefault(node, "dilations", {1, 1});

  // Normalize padding format
  // ONNX can have [pad_h, pad_w] or [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
  if (pads.size() == 2) {
    pads = {pads[0], pads[1], pads[0], pads[1]};
  } else if (pads.size() != 4) {
    return Status::Failure("Conv pads must have 2 or 4 elements");
  }

  // Determine compute data type from input data types
  auto compute_dtype = GetComputeDataType(x_attr->get_data_type(), w_attr->get_data_type());
  if (!compute_dtype.has_value()) {
    return Status::Failure("Unsupported data type combination for Conv compute");
  }

  // Create convolution attributes
  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding({pads[0], pads[1]})  // Use begin padding
      .set_stride({strides[0], strides[1]})
      .set_dilation({dilations[0], dilations[1]})
      .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
      .set_compute_data_type(compute_dtype.value());

  // Add convolution to graph - returns output tensor attributes
  output_attr = graph.conv_fprop(x_attr, w_attr, conv_attrs);

  // Optional bias: B may be a scalar or a 1D [C_out] tensor.
  // Both are handled via pointwise ADD which broadcasts over the conv output.
  if (input_attrs.size() >= 3) {
    auto dtype = compute_dtype.value();
    output_attr->set_data_type(dtype);
    auto bias = input_attrs[2];
    auto reshape_status = ReshapeBiasForConv(bias, layout);
    if (reshape_status.failed()) return reshape_status;

    PointwiseAttributes add;
    add.set_mode(PointwiseMode::ADD).set_compute_data_type(dtype);
    output_attr = graph.pointwise(output_attr, bias, add);
  }

  // Set output dim and stride so that callers (Build()) do not need to know
  // the convolution layout.  For NHWC, relabel ORT's [N,H,W,C] to hipDNN's
  // [N,C,H,W] and apply NHWC strides; for NCHW, read the shape as-is and
  // apply row-major strides.
  {
    std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
    if (!outputs.empty()) {
      auto shape = GetTensorShape(outputs[0]);
      if (shape.has_value() && shape->size() == 4) {
        if (layout == ConvLayout::NHWC) {
          // ORT shape [N, H, W, C] -> hipDNN dim [N, C, H, W]
          std::vector<int64_t> out_dim = {
              (*shape)[0], (*shape)[3], (*shape)[1], (*shape)[2]};
          output_attr->set_dim(out_dim);
          output_attr->set_stride(ComputeNHWCStrides(out_dim));
        } else {
          output_attr->set_dim(shape.value());
          output_attr->set_stride(ComputeStrides(shape.value()));
        }
      }
    }
  }

  return Status::Success();
}

// Apply a permutation to a tensor's dims and strides.
// This is a zero-copy operation — same data buffer, different view.
// `perm` maps new axis i to old axis perm[i], e.g. {1, 0} transposes a 2D
// tensor and {0, 2, 1} swaps the last two axes of a 3D tensor.
static void PermuteTensorAttr(TensorAttrPtr& attr,
                              const std::vector<int64_t>& perm) {
  auto dims = attr->get_dim();
  auto strides = attr->get_stride();
  std::vector<int64_t> new_dims(perm.size());
  std::vector<int64_t> new_strides(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    new_dims[i] = dims[perm[i]];
    new_strides[i] = strides[perm[i]];
  }
  attr->set_dim(new_dims).set_stride(new_strides);
}

// Add MatMul/Gemm operation to hipDNN graph
// MatMul: Y = A @ B
// Gemm: Y = alpha * A' * B' + beta * C
//   Transpose via stride manipulation, alpha/beta via pointwise ops
Status AddMatMulNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr,
    int64_t& next_uid) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::PointwiseMode;

  if (input_attrs.size() < 2) {
    return Status::Failure("MatMul/Gemm requires at least 2 input tensor attributes");
  }

  auto a_attr = input_attrs[0];
  auto b_attr = input_attrs[1];

  // A and B must be tensors, not scalars.
  if (IsScalarAttr(a_attr)) {
    return Status::Failure("MatMul/Gemm input A must be a tensor, not a scalar");
  }
  if (IsScalarAttr(b_attr)) {
    return Status::Failure("MatMul/Gemm input B must be a tensor, not a scalar");
  }

  // Gemm-specific attributes (defaults match MatMul semantics)
  int64_t trans_a = GetIntAttrOrDefault(node, "transA", 0);
  int64_t trans_b = GetIntAttrOrDefault(node, "transB", 0);
  float alpha = GetFloatAttrOrDefault(node, "alpha", 1.0f);
  float beta = GetFloatAttrOrDefault(node, "beta", 1.0f);

  // Handle transpose by permuting dims and strides (zero-copy).
  // Gemm transA/transB swap the last two axes.
  if (trans_a != 0 || trans_b != 0) {
    size_t rank = a_attr->get_dim().size();
    std::vector<int64_t> perm(rank);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[rank - 2], perm[rank - 1]);
    if (trans_a != 0) {
      PermuteTensorAttr(a_attr, perm);
    }
    if (trans_b != 0) {
      PermuteTensorAttr(b_attr, perm);
    }
  }

  auto compute_dtype = GetComputeDataType(a_attr->get_data_type(), b_attr->get_data_type());
  if (!compute_dtype.has_value()) {
    return Status::Failure("Unsupported data type combination for MatMul/Gemm compute");
  }

  // matmul: result = A @ B
  MatmulAttributes matmul_attrs;
  matmul_attrs.set_compute_data_type(compute_dtype.value());
  auto result = graph.matmul(a_attr, b_attr, matmul_attrs);

  // When chaining ops, intermediate virtual tensors need their data type set
  // (Build() only sets properties on the final graph output).
  auto dtype = compute_dtype.value();
  auto set_intermediate_dtype = [&](TensorAttrPtr& t) {
    t->set_data_type(dtype);
  };

  // alpha scaling: result = alpha * (A @ B)
  if (alpha != 1.0f) {
    set_intermediate_dtype(result);
    auto alpha_attr = CreateScalarTensorAttr(next_uid++, alpha);
    PointwiseAttributes pw;
    pw.set_mode(PointwiseMode::MUL).set_compute_data_type(dtype);
    result = graph.pointwise(result, alpha_attr, pw);
  }

  // bias: result = result + beta * C
  bool has_bias = (beta != 0.0f && input_attrs.size() >= 3);
  if (has_bias) {
    set_intermediate_dtype(result);
    auto bias = input_attrs[2];

    // Scale bias if beta != 1.0
    if (beta != 1.0f) {
      auto beta_attr = CreateScalarTensorAttr(next_uid++, beta);
      PointwiseAttributes pw;
      pw.set_mode(PointwiseMode::MUL).set_compute_data_type(dtype);
      bias = graph.pointwise(bias, beta_attr, pw);
      set_intermediate_dtype(bias);
    }

    PointwiseAttributes add;
    add.set_mode(PointwiseMode::ADD).set_compute_data_type(dtype);
    result = graph.pointwise(result, bias, add);
  }

  output_attr = result;
  return Status::Success();
}

// Returns true if the ONNX op type is handled by the Fusilli engine.
// When all ops in a graph are Fusilli-compatible we request that engine;
// otherwise we let hipDNN pick the best available engine per-op.
// Keep in sync with IsFusilliCompatibleMLIROp below.
// MatMul/Gemm are included because hipBLAS-LT is currently disabled;
// revisit when re-enabled.
// Note: MultiHeadAttention (SDPA) is intentionally absent — it uses
// hipDNN's dedicated SDPA engine, not Fusilli.
bool IsFusilliCompatibleOp(const std::string& op_type) {
  return op_type == "Conv" || op_type == "MatMul" || op_type == "Gemm" ||
         op_type == "Mul" || op_type == "Sub" || op_type == "Add" ||
         op_type == "Div";
}

// Map ONNX pointwise op name to hipDNN PointwiseMode.
// Keep this list in sync with the pointwise dispatch in
// src/core/ep.cc (IsSupportedOp).
static std::optional<hipdnn_frontend::PointwiseMode> GetPointwiseMode(
    const std::string& op_type) {
  using hipdnn_frontend::PointwiseMode;
  if (op_type == "Mul") return PointwiseMode::MUL;
  if (op_type == "Sub") return PointwiseMode::SUB;
  if (op_type == "Add") return PointwiseMode::ADD;
  if (op_type == "Div") return PointwiseMode::DIV;
  return std::nullopt;
}

// Map canonical unary pointwise op name to hipDNN PointwiseMode.
// This is the single source of truth for unary pointwise dispatch;
// the MLIR path strips the "torch.aten." prefix before calling this.
// Keep the op names in sync with ep.cc (IsSupportedOp).
static std::optional<hipdnn_frontend::PointwiseMode> GetUnaryPointwiseMode(
    const std::string& op_name) {
  using hipdnn_frontend::PointwiseMode;
  if (op_name == "Sigmoid") return PointwiseMode::SIGMOID_FWD;
  return std::nullopt;
}

// Add a pointwise binary operation (Mul, Sub, Add, Div) to hipDNN graph
// Takes two input tensor attributes, returns output tensor attribute
Status AddPointwiseNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  using namespace hipdnn_frontend::graph;

  if (input_attrs.size() != 2) {
    return Status::Failure("Pointwise binary op requires exactly 2 input tensor attributes");
  }

  std::string op_type = node.GetOperatorType();
  auto mode = GetPointwiseMode(op_type);
  if (!mode.has_value()) {
    return Status::Failure("Unsupported pointwise op type: " + op_type);
  }

  const auto& a_attr = input_attrs[0];
  const auto& b_attr = input_attrs[1];

  // Pointwise ops have no reduction, so compute_data_type just needs to
  // match the input precision.  Use the first operand's type.
  PointwiseAttributes pw;
  pw.set_mode(mode.value()).set_compute_data_type(a_attr->get_data_type());
  output_attr = graph.pointwise(a_attr, b_attr, pw);

  return Status::Success();
}

// Add a unary pointwise operation (Sigmoid) to hipDNN graph
// Takes one input tensor attribute, returns output tensor attribute
static Status AddUnaryPointwiseNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  using namespace hipdnn_frontend::graph;

  if (input_attrs.size() != 1) {
    return Status::Failure("Unary pointwise op requires exactly 1 input tensor attribute");
  }

  std::string op_type = node.GetOperatorType();
  auto mode = GetUnaryPointwiseMode(op_type);
  if (!mode.has_value()) {
    return Status::Failure("Unsupported unary pointwise op type: " + op_type);
  }

  const auto& x_attr = input_attrs[0];

  // Pointwise ops have no reduction, so compute_data_type just needs to
  // match the input precision.
  PointwiseAttributes pw;
  pw.set_mode(mode.value()).set_compute_data_type(x_attr->get_data_type());
  output_attr = graph.pointwise(x_attr, pw);

  return Status::Success();
}

// Add MultiHeadAttention or GroupQueryAttention (SDPA) operation to hipDNN graph.
//
// Both ops take Q, K, V in [B, S, hidden_size] format.  For MHA, all three
// use num_heads * head_size.  For GQA, K/V use kv_num_heads * head_size
// (fewer heads, shared across query head groups).
//
// hipDNN's graph.sdpa() expects [B, H, S, D] and natively supports different
// head counts for Q vs K/V.  We reshape via stride manipulation (zero-copy):
//
//   [B, S, H*D] row-major strides: [S*H*D, H*D, 1]
//   viewed as [B, H, S, D] with strides: [S*H*D, D, H*D, 1]
//
// The output from SDPA is [B, H_q, S, D] and needs to be stored in
// [B, S, H_q*D] order.  We set the output tensor's strides to
// [S*H_q*D, D, H_q*D, 1], which tells hipDNN to write the output in an order
// that, when interpreted as contiguous [B, S, H_q*D], is correct.
static Status AddSdpaNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr,
    int64_t& next_uid) {
  using namespace hipdnn_frontend::graph;

  // We need at least Q, K, V.
  if (input_attrs.size() < 3) {
    return Status::Failure("SDPA requires at least 3 inputs (Q, K, V)");
  }

  // Reshape the input tensor attrs in place.  This is safe because the EP
  // currently claims nodes individually (no fusion), so each tensor attr is
  // consumed by exactly one node.  The attrs must be the same shared_ptr
  // objects that Build() marked as is_virtual(false); passing copies would
  // disconnect them from the graph's input tracking and hipDNN would treat
  // them as virtual (internal) tensors with no data pointers.
  auto q_attr = input_attrs[0];
  auto k_attr = input_attrs[1];
  auto v_attr = input_attrs[2];

  // Q, K, V must be 3D tensors.
  if (q_attr->get_dim().size() != 3 || k_attr->get_dim().size() != 3 ||
      v_attr->get_dim().size() != 3) {
    return Status::Failure("SDPA inputs must be 3D [B, S, hidden_size]");
  }

  int64_t num_heads = GetIntAttrOrDefault(node, "num_heads", 0);
  if (num_heads <= 0) {
    return Status::Failure("SDPA requires num_heads > 0");
  }

  // For GQA, kv_num_heads may differ from num_heads.  For MHA, K/V use the
  // same num_heads as Q (kv_num_heads attribute absent or 0).
  int64_t kv_num_heads = GetIntAttrOrDefault(node, "kv_num_heads", 0);
  if (kv_num_heads <= 0) {
    kv_num_heads = num_heads;
  }

  auto q_dim = q_attr->get_dim();
  int64_t batch_size = q_dim[0];
  int64_t seq_len_q = q_dim[1];
  int64_t q_hidden = q_dim[2];
  int64_t head_size = q_hidden / num_heads;

  auto k_dim = k_attr->get_dim();
  int64_t seq_len_kv = k_dim[1];
  int64_t kv_hidden = kv_num_heads * head_size;

  // Reshape Q from [B, S, H_q*D] to [B, H_q, S, D] via stride manipulation.
  q_attr->set_dim({batch_size, num_heads, seq_len_q, head_size});
  q_attr->set_stride({seq_len_q * q_hidden, head_size, q_hidden, 1});

  // Reshape K from [B, S, H_kv*D] to [B, H_kv, S, D].
  k_attr->set_dim({batch_size, kv_num_heads, seq_len_kv, head_size});
  k_attr->set_stride({seq_len_kv * kv_hidden, head_size, kv_hidden, 1});

  // Reshape V from [B, S, H_kv*D] to [B, H_kv, S, D].
  v_attr->set_dim({batch_size, kv_num_heads, seq_len_kv, head_size});
  v_attr->set_stride({seq_len_kv * kv_hidden, head_size, kv_hidden, 1});

  // Determine compute data type.
  auto compute_dtype =
      GetComputeDataType(q_attr->get_data_type(), k_attr->get_data_type());
  if (!compute_dtype.has_value()) {
    return Status::Failure("Unsupported data type combination for SDPA compute");
  }

  // Build SdpaAttributes.
  SdpaAttributes sdpa_attrs;
  sdpa_attrs.set_compute_data_type(compute_dtype.value());

  // Scale: default is 1/sqrt(head_size), can be overridden by attribute.
  float scale = GetFloatAttrOrDefault(node, "scale", 0.0f);
  if (scale != 0.0f) {
    sdpa_attrs.attn_scale_value = scale;
  } else {
    sdpa_attrs.attn_scale_value =
        1.0f / std::sqrt(static_cast<float>(head_size));
  }

  // Causal masking: unidirectional=1 means causal.
  int64_t unidirectional = GetIntAttrOrDefault(node, "unidirectional", 0);
  if (unidirectional != 0) {
    sdpa_attrs.causal_mask = true;
  }

  // No dropout for inference.
  // No stats generation needed (inference only).
  // Attention bias is not yet supported — requires verifying how ORT
  // maps absent optional inputs in the fused graph.

  // Call graph.sdpa() — returns [output, stats].
  auto [o_attr, stats_attr] = graph.sdpa(q_attr, k_attr, v_attr, sdpa_attrs);

  // Reshape output from [B, H_q, S_q, D] to [B, S_q, H_q*D] via strides.
  // Output always uses num_heads (Q heads), not kv_num_heads.
  o_attr->set_dim({batch_size, num_heads, seq_len_q, head_size});
  o_attr->set_stride({seq_len_q * q_hidden, head_size, q_hidden, 1});

  output_attr = o_attr;

  return Status::Success();
}

// Add SimplifiedLayerNormalization (RMS Norm) to hipDNN graph.
// Inputs: X, Scale.  Outputs: Y (and optionally inv_std_var).
//
// hipDNN's RMSNorm always normalizes over axis 1 (channel) and expects
// Scale with shape [1, C, 1, 1].  ONNX's SimplifiedLayerNormalization
// normalizes over dimensions [axis:rank] with Scale shape matching those
// dims.  To bridge this gap, we reshape:
//   X: [d0, d1, ..., d_{axis-1}, d_axis, ..., d_{rank-1}]
//     -> [1, C, N, 1]  where N = product(d0..d_{axis-1}), C = product(d_axis..d_{rank-1})
//   Scale: [d_axis, ..., d_{rank-1}]
//     -> [1, C, 1, 1]
//   Y output: [1, C, N, 1] (caller's Build loop sets final shape from ORT graph)
// The batch dimensions are placed in the spatial (H) position so that
// hipDNN normalizes over C for each independent batch position.
static Status AddRMSNormNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    std::vector<TensorAttrPtr>& output_attrs,
    int64_t& next_uid) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::NormFwdPhase;

  if (input_attrs.size() != 2) {
    return Status::Failure("SimplifiedLayerNormalization requires exactly 2 inputs (X, Scale)");
  }

  auto x_attr = input_attrs[0];
  auto scale_attr = input_attrs[1];

  if (IsScalarAttr(x_attr)) {
    return Status::Failure("SimplifiedLayerNormalization input X must be a tensor, not a scalar");
  }

  // Determine whether the optional inv_std_var output is requested.
  // In ONNX, optional outputs may appear in the list with an empty name
  // to indicate they are not requested.
  std::vector<Ort::ConstValueInfo> node_outputs = node.GetOutputs();
  bool need_inv_std_var =
      (node_outputs.size() > 1 && !node_outputs[1].GetName().empty());

  // Resolve the axis attribute.
  auto x_dims = x_attr->get_dim();
  int64_t rank = static_cast<int64_t>(x_dims.size());
  int64_t axis = GetIntAttrOrDefault(node, "axis", -1);
  if (axis < 0) {
    axis += rank;
  }

  // Compute the batch (N) and channel (C) sizes for hipDNN's view.
  int64_t batch_size = 1;
  for (int64_t i = 0; i < axis; ++i) {
    batch_size *= x_dims[i];
  }
  int64_t channel_size = 1;
  for (int64_t i = axis; i < rank; ++i) {
    channel_size *= x_dims[i];
  }

  // Reshape X to [1, C, N, 1] for hipDNN RMSNorm (normalizes over axis 1).
  // Batch dims go into the spatial (H) position so hipDNN normalizes over C
  // independently for each batch element.
  x_attr->set_dim({1, channel_size, batch_size, 1});
  x_attr->set_stride(ComputeStrides({1, channel_size, batch_size, 1}));

  // Reshape Scale to [1, C, 1, 1] (per-channel scaling).
  scale_attr->set_dim({1, channel_size, 1, 1});
  scale_attr->set_stride(ComputeStrides({1, channel_size, 1, 1}));

  // Extract epsilon attribute (default 1e-5 per ONNX spec).
  float epsilon = GetFloatAttrOrDefault(node, "epsilon", 1e-5f);
  auto epsilon_attr = CreateScalarTensorAttr(next_uid++, epsilon);

  // Build RMSNorm attributes.  Use TRAINING phase when the optional
  // inv_std_var output is needed (hipDNN only produces it in that mode).
  RMSNormAttributes rmsnorm_attrs;
  rmsnorm_attrs
      .set_forward_phase(need_inv_std_var ? NormFwdPhase::TRAINING : NormFwdPhase::INFERENCE)
      .set_epsilon(epsilon_attr)
      .set_compute_data_type(x_attr->get_data_type());

  // Call graph.rmsnorm() which returns [y, invRms].
  // invRms is nullptr in INFERENCE mode.
  auto [y_attr, inv_rms_attr] = graph.rmsnorm(x_attr, scale_attr, rmsnorm_attrs);

  // Set Y shape to [1, C, N, 1] matching the reshaped X so hipDNN graph
  // validation passes.  The caller's Build loop preserves this (dim is
  // non-empty), but Execute uses output_shapes_ (the original ORT shape)
  // for output tensor allocation, so the reshape is transparent to ORT.
  y_attr->set_dim({1, channel_size, batch_size, 1});
  y_attr->set_stride(ComputeStrides({1, channel_size, batch_size, 1}));
  output_attrs.push_back(y_attr);

  if (need_inv_std_var) {
    assert(inv_rms_attr != nullptr && "TRAINING phase must produce inv_rms output");
    output_attrs.push_back(inv_rms_attr);
  }
  return Status::Success();
}

// Dispatch to appropriate Add*Node based on op_type.
// Takes input tensor attributes, returns output tensor attributes.
Status AddNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    std::vector<TensorAttrPtr>& output_attrs,
    int64_t& next_uid) {
  std::string op_type = node.GetOperatorType();

  if (op_type == "Conv") {
    TensorAttrPtr y_attr;
    auto status = AddConvNode(graph, node, input_attrs, y_attr, next_uid);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  if (op_type == "MatMul" || op_type == "Gemm") {
    TensorAttrPtr y_attr;
    auto status = AddMatMulNode(
        graph, node, input_attrs, y_attr, next_uid);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  if (GetPointwiseMode(op_type).has_value()) {
    TensorAttrPtr y_attr;
    auto status = AddPointwiseNode(graph, node, input_attrs, y_attr);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  if (GetUnaryPointwiseMode(op_type).has_value()) {
    TensorAttrPtr y_attr;
    auto status = AddUnaryPointwiseNode(graph, node, input_attrs, y_attr);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  if (op_type == "MultiHeadAttention" || op_type == "GroupQueryAttention") {
    TensorAttrPtr y_attr;
    auto status = AddSdpaNode(graph, node, input_attrs, y_attr, next_uid);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  if (op_type == "SimplifiedLayerNormalization") {
    return AddRMSNormNode(graph, node, input_attrs, output_attrs, next_uid);
  }

  return Status::Failure("Unsupported op type: " + op_type);
}

#ifdef HIPDNN_EP_HAS_TORCH_MLIR

// Result struct for GetTensorInfo
struct TensorInfo {
  std::vector<int64_t> shape;
  hipdnn_frontend::DataType dtype;
};

// Convert MLIR element type to hipDNN data type
mlir::FailureOr<hipdnn_frontend::DataType> MLIRTypeToHipDNNDataType(
    mlir::Location loc,
    mlir::Type type) {
  using hipdnn_frontend::DataType;
  if (type.isF32()) {
    return DataType::FLOAT;
  }
  if (type.isF16()) {
    return DataType::HALF;
  }
  return mlir::emitError(loc) << "unsupported element type: " << type;
}

// Extract shape and element type from torch.vtensor type
mlir::FailureOr<TensorInfo> GetTensorInfo(mlir::Location loc, mlir::Type type) {
  auto vtensor = mlir::dyn_cast<mlir::torch::Torch::ValueTensorType>(type);
  if (!vtensor) {
    return mlir::emitError(loc) << "expected torch.vtensor type, got: " << type;
  }
  if (!vtensor.hasSizes()) {
    return mlir::emitError(loc) << "vtensor type has no static shape";
  }
  if (!vtensor.hasDtype()) {
    return mlir::emitError(loc) << "vtensor type has no dtype";
  }

  TensorInfo info;
  auto sizes = vtensor.getSizes();
  info.shape.assign(sizes.begin(), sizes.end());

  auto dtype = MLIRTypeToHipDNNDataType(loc, vtensor.getDtype());
  if (mlir::failed(dtype)) {
    return mlir::failure();
  }
  info.dtype = *dtype;
  return info;
}

// Create TensorAttributes from MLIR type.
// Strides are always row-major; NHWC strides are applied by the node builder.
mlir::FailureOr<TensorAttrPtr> CreateTensorAttrFromMLIR(
    mlir::Location loc,
    mlir::Type type,
    int64_t uid,
    const std::string& name) {
  using hipdnn_frontend::graph::TensorAttributes;

  auto info = GetTensorInfo(loc, type);
  if (mlir::failed(info)) {
    return mlir::failure();
  }

  auto attr = std::make_shared<TensorAttributes>();
  attr->set_uid(uid)
      .set_name(name)
      .set_data_type(info->dtype)
      .set_dim(info->shape)
      .set_stride(ComputeStrides(info->shape));

  return attr;
}

// Add Conv operation from MLIR op to hipDNN graph
Status AddConvNodeFromMLIR(hipdnn_frontend::graph::Graph& graph,
                           mlir::Operation* op,
                           const std::vector<TensorAttrPtr>& input_attrs,
                           TensorAttrPtr& output_attr,
                           int64_t& next_uid) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::ConvolutionMode;
  using hipdnn_frontend::PointwiseMode;

  if (input_attrs.size() < 2) {
    return Status::Failure("Conv requires at least 2 input tensor attributes");
  }

  const auto& x_attr = input_attrs[0];
  const auto& w_attr = input_attrs[1];

  // X and W must be tensors, not scalars.
  if (IsScalarAttr(x_attr)) {
    return Status::Failure("Conv input X must be a tensor, not a scalar");
  }
  if (IsScalarAttr(w_attr)) {
    return Status::Failure("Conv filter W must be a tensor, not a scalar");
  }

  // The MLIR path only supports NCHW convolutions.  NHWC is rejected at
  // module build time (IRBuilderImpl::BuildModule) so it cannot reach here.
  ConvLayout layout = ConvLayout::NCHW;

  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> dilations = {1, 1};

  if (auto convOp = mlir::dyn_cast<mlir::torch::Torch::AtenConvolutionOp>(op)) {
    llvm::SmallVector<int64_t> strideVals;
    if (mlir::matchPattern(
            convOp.getStride(),
            mlir::torch::Torch::m_TorchListOfConstantInts(strideVals))) {
      strides.assign(strideVals.begin(), strideVals.end());
    }

    llvm::SmallVector<int64_t> padVals;
    if (mlir::matchPattern(
            convOp.getPadding(),
            mlir::torch::Torch::m_TorchListOfConstantInts(padVals))) {
      if (padVals.size() == 2) {
        pads = {padVals[0], padVals[1], padVals[0], padVals[1]};
      }
    }

    llvm::SmallVector<int64_t> dilationVals;
    if (mlir::matchPattern(
            convOp.getDilation(),
            mlir::torch::Torch::m_TorchListOfConstantInts(dilationVals))) {
      dilations.assign(dilationVals.begin(), dilationVals.end());
    }
  }

  auto compute_dtype =
      GetComputeDataType(x_attr->get_data_type(), w_attr->get_data_type());
  if (!compute_dtype.has_value()) {
    return Status::Failure("Unsupported data type combination for Conv compute");
  }

  ConvFpropAttributes conv_attrs;
  conv_attrs.set_padding({pads[0], pads[1]})
      .set_stride({strides[0], strides[1]})
      .set_dilation({dilations[0], dilations[1]})
      .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION)
      .set_compute_data_type(compute_dtype.value());

  output_attr = graph.conv_fprop(x_attr, w_attr, conv_attrs);

  // Optional bias: may be a scalar or a 1D [C_out] tensor.
  if (input_attrs.size() >= 3) {
    auto dtype = compute_dtype.value();
    output_attr->set_data_type(dtype);
    auto bias = input_attrs[2];
    auto reshape_status = ReshapeBiasForConv(bias, layout);
    if (reshape_status.failed()) return reshape_status;

    PointwiseAttributes add;
    add.set_mode(PointwiseMode::ADD).set_compute_data_type(dtype);
    output_attr = graph.pointwise(output_attr, bias, add);
  }

  return Status::Success();
}

// Add MatMul/Gemm operation from MLIR op to hipDNN graph
// Handles torch.aten.mm, torch.aten.matmul (simple matmul)
// and torch.aten.addmm (Y = alpha * mat1 @ mat2 + beta * self)
Status AddMatMulNodeFromMLIR(hipdnn_frontend::graph::Graph& graph,
                             mlir::Operation* op,
                             const std::vector<TensorAttrPtr>& input_attrs,
                             TensorAttrPtr& output_attr,
                             int64_t& next_uid) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::PointwiseMode;

  // Determine which op we're handling and extract the A, B tensors
  // and optional bias/alpha/beta.
  TensorAttrPtr a_attr, b_attr;
  TensorAttrPtr bias_attr;
  float alpha = 1.0f;
  float beta = 1.0f;

  if (auto addmmOp = mlir::dyn_cast<mlir::torch::Torch::AtenAddmmOp>(op)) {
    // addmm: result = alpha * mat1 @ mat2 + beta * self
    // input_attrs: [self(bias), mat1, mat2]
    if (input_attrs.size() < 3) {
      return Status::Failure("addmm requires 3 tensor inputs");
    }
    bias_attr = input_attrs[0];
    a_attr = input_attrs[1];
    b_attr = input_attrs[2];

    // Extract alpha and beta scalars
    double alpha_val, beta_val;
    if (mlir::matchPattern(addmmOp.getAlpha(),
                           mlir::torch::Torch::m_TorchConstantFloat(&alpha_val))) {
      alpha = static_cast<float>(alpha_val);
    } else {
      int64_t alpha_int;
      if (mlir::matchPattern(addmmOp.getAlpha(),
                             mlir::torch::Torch::m_TorchConstantInt(&alpha_int))) {
        alpha = static_cast<float>(alpha_int);
      }
    }
    if (mlir::matchPattern(addmmOp.getBeta(),
                           mlir::torch::Torch::m_TorchConstantFloat(&beta_val))) {
      beta = static_cast<float>(beta_val);
    } else {
      int64_t beta_int;
      if (mlir::matchPattern(addmmOp.getBeta(),
                             mlir::torch::Torch::m_TorchConstantInt(&beta_int))) {
        beta = static_cast<float>(beta_int);
      }
    }
  } else {
    // mm / matmul: result = A @ B
    if (input_attrs.size() < 2) {
      return Status::Failure("mm/matmul requires 2 tensor inputs");
    }
    a_attr = input_attrs[0];
    b_attr = input_attrs[1];
  }

  // A and B must be tensors, not scalars.
  if (IsScalarAttr(a_attr)) {
    return Status::Failure("MatMul input A must be a tensor, not a scalar");
  }
  if (IsScalarAttr(b_attr)) {
    return Status::Failure("MatMul input B must be a tensor, not a scalar");
  }

  auto compute_dtype =
      GetComputeDataType(a_attr->get_data_type(), b_attr->get_data_type());
  if (!compute_dtype.has_value()) {
    return Status::Failure(
        "Unsupported data type combination for MatMul compute");
  }

  // matmul: result = A @ B
  MatmulAttributes matmul_attrs;
  matmul_attrs.set_compute_data_type(compute_dtype.value());
  auto result = graph.matmul(a_attr, b_attr, matmul_attrs);

  auto dtype = compute_dtype.value();
  auto set_intermediate_dtype = [&](TensorAttrPtr& t) {
    t->set_data_type(dtype);
  };

  // alpha scaling: result = alpha * (A @ B)
  if (alpha != 1.0f) {
    set_intermediate_dtype(result);
    auto alpha_attr =
        CreateScalarTensorAttr(next_uid++, alpha);
    PointwiseAttributes pw;
    pw.set_mode(PointwiseMode::MUL).set_compute_data_type(dtype);
    result = graph.pointwise(result, alpha_attr, pw);
  }

  // bias: result = result + beta * bias
  if (bias_attr && beta != 0.0f) {
    set_intermediate_dtype(result);
    auto bias = bias_attr;

    if (beta != 1.0f) {
      auto beta_scalar =
          CreateScalarTensorAttr(next_uid++, beta);
      PointwiseAttributes pw;
      pw.set_mode(PointwiseMode::MUL).set_compute_data_type(dtype);
      bias = graph.pointwise(bias, beta_scalar, pw);
      set_intermediate_dtype(bias);
    }

    PointwiseAttributes add;
    add.set_mode(PointwiseMode::ADD).set_compute_data_type(dtype);
    result = graph.pointwise(result, bias, add);
  }

  output_attr = result;
  return Status::Success();
}

// Returns true if the MLIR op is handled by the Fusilli engine.
// Keep in sync with IsFusilliCompatibleOp above.
// Pointwise MLIR ops are not yet handled by AddNodeFromMLIR; extend
// here when added.
bool IsFusilliCompatibleMLIROp(llvm::StringRef op_name) {
  return op_name == "torch.aten.convolution" ||
         op_name == "torch.aten.conv2d" || op_name == "torch.aten.mm" ||
         op_name == "torch.aten.matmul" || op_name == "torch.aten.addmm";
}

// Map torch.aten unary op name to hipDNN PointwiseMode by stripping
// the "torch.aten." prefix and delegating to GetUnaryPointwiseMode.
static std::optional<hipdnn_frontend::PointwiseMode> GetUnaryPointwiseModeFromMLIR(
    llvm::StringRef op_name) {
  if (!op_name.consume_front("torch.aten.")) return std::nullopt;
  // Capitalize the first letter to match ONNX op names (e.g., "sigmoid" -> "Sigmoid").
  std::string canonical = op_name.str();
  if (!canonical.empty()) {
    canonical[0] = std::toupper(static_cast<unsigned char>(canonical[0]));
  }
  return GetUnaryPointwiseMode(canonical);
}

// Add a unary pointwise operation from MLIR (e.g., torch.aten.sigmoid)
static Status AddUnaryPointwiseNodeFromMLIR(
    hipdnn_frontend::graph::Graph& graph,
    mlir::Operation* op,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  using namespace hipdnn_frontend::graph;

  if (input_attrs.size() != 1) {
    return Status::Failure("MLIR unary pointwise op requires exactly 1 input tensor attribute");
  }

  auto mode = GetUnaryPointwiseModeFromMLIR(op->getName().getStringRef());
  if (!mode.has_value()) {
    return Status::Failure("Unsupported MLIR unary pointwise op: " +
                           op->getName().getStringRef().str());
  }

  const auto& x_attr = input_attrs[0];

  // Pointwise ops have no reduction, so compute_data_type just needs to
  // match the input precision.
  PointwiseAttributes pw;
  pw.set_mode(mode.value()).set_compute_data_type(x_attr->get_data_type());
  output_attr = graph.pointwise(x_attr, pw);

  return Status::Success();
}

// Dispatch MLIR op to appropriate Add*Node function.
Status AddNodeFromMLIR(hipdnn_frontend::graph::Graph& graph,
                       mlir::Operation* op,
                       const std::vector<TensorAttrPtr>& input_attrs,
                       std::vector<TensorAttrPtr>& output_attrs,
                       int64_t& next_uid) {
  llvm::StringRef op_name = op->getName().getStringRef();

  if (op_name == "torch.aten.convolution" || op_name == "torch.aten.conv2d") {
    TensorAttrPtr y_attr;
    auto status = AddConvNodeFromMLIR(graph, op, input_attrs, y_attr,
                                      next_uid);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  if (op_name == "torch.aten.mm" || op_name == "torch.aten.matmul" ||
      op_name == "torch.aten.addmm") {
    TensorAttrPtr y_attr;
    auto status = AddMatMulNodeFromMLIR(graph, op, input_attrs, y_attr,
                                        next_uid);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  if (GetUnaryPointwiseModeFromMLIR(op_name).has_value()) {
    TensorAttrPtr y_attr;
    auto status = AddUnaryPointwiseNodeFromMLIR(graph, op, input_attrs, y_attr);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
  }

  return Status::Failure("Unsupported MLIR op type: " + op_name.str());
}

#endif  // HIPDNN_EP_HAS_TORCH_MLIR

}  // namespace

//
// HipDNNGraphImpl - pimpl implementation
//

struct HipDNNGraphImpl {
  explicit HipDNNGraphImpl(hipdnnHandle_t handle) : handle_(handle) {}

  Status Build(const std::vector<Ort::ConstValueInfo>& graph_inputs,
               const std::vector<Ort::ConstValueInfo>& graph_outputs,
               const std::vector<Ort::ConstNode>& nodes);

#ifdef HIPDNN_EP_HAS_TORCH_MLIR
  Status Build(mlir::Region& region);
#endif

  Status Compile();

  Status Execute(OrtKernelContext* kernel_ctx);

 private:
  hipdnnHandle_t handle_;

  // hipDNN graph
  std::unique_ptr<hipdnn_frontend::graph::Graph> graph_;

  // Workspace for hipDNN graph
  std::vector<char> workspace_;

  // Graph input/output info.
  // input_uids_ maps each ORT input index to a hipDNN UID.  Embedded scalar
  // constants get kEmbeddedScalar (-1) — they have no runtime data pointer.
  static constexpr int64_t kEmbeddedScalar = -1;
  std::vector<int64_t> input_uids_;
  std::vector<int64_t> output_uids_;
  std::vector<std::vector<int64_t>> output_shapes_;

  // Symbol table: maps value name to TensorAttributes
  std::unordered_map<std::string, TensorAttrPtr> symbol_table_;

  // UID counter for tensor attributes
  int64_t next_uid_{1};

  // True when every op in the graph is handled by the Fusilli engine.
  // Set during Build(); Compile() uses it to decide whether to request
  // the FUSILLI_ENGINE or let hipDNN choose per-op.
  bool all_fusilli_compatible_{true};
};

Status HipDNNGraphImpl::Build(
    const std::vector<Ort::ConstValueInfo>& graph_inputs,
    const std::vector<Ort::ConstValueInfo>& graph_outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  using namespace hipdnn_frontend::graph;

  // Store output shapes for Execute
  output_shapes_.reserve(graph_outputs.size());
  for (const auto& output : graph_outputs) {
    auto shape = GetTensorShape(output);
    if (!shape.has_value()) {
      return Status::Failure("Graph output must have static shape: " + output.GetName());
    }
    output_shapes_.push_back(shape.value());
  }

  graph_ = std::make_unique<Graph>();

  // Set graph-level data types from the first graph input.  These are needed
  // by ops like SDPA and RMSNorm whose engine lookup depends on graph-level
  // types.  Some ops (e.g., RMSNorm) use fill_from_context to propagate data
  // types to internally-created tensors; without graph-level defaults,
  // engine heuristics may fail to find a matching configuration.
  // For ops that set their own compute_data_type (Conv, pointwise, etc.)
  // the per-op type takes precedence.
  if (!graph_inputs.empty()) {
    auto first_dtype = ToHipDNNDataType(GetTensorElementType(graph_inputs[0]));
    if (first_dtype.has_value()) {
      graph_->set_io_data_type(first_dtype.value())
          .set_intermediate_data_type(first_dtype.value())
          .set_compute_data_type(first_dtype.value());
    }
  }

  // Create TensorAttributes for all graph inputs and add to symbol table.
  // Scalar constant initializers (element count == 1) are embedded directly
  // into the graph via the pass-by-value API instead of becoming runtime
  // inputs.  This avoids requiring the caller to provide a device pointer
  // for values that are known at graph-build time.
  input_uids_.reserve(graph_inputs.size());
  for (const auto& input : graph_inputs) {
    if (auto attr = TryExtractScalarConstant(input)) {
      // Embed the scalar directly via pass-by-value — no runtime input needed.
      attr->set_uid(next_uid_++).set_name(input.GetName());
      symbol_table_[input.GetName()] = attr;
      input_uids_.push_back(kEmbeddedScalar);
      continue;
    }

    TensorAttrPtr attr;
    auto status = CreateTensorAttr(input, next_uid_++, attr);
    if (status.failed()) return status;
    attr->set_is_virtual(false);
    symbol_table_[input.GetName()] = attr;
    input_uids_.push_back(attr->get_uid());
  }

  // Process each node in the graph
  for (const auto& node : nodes) {
    // Track whether all ops are Fusilli-compatible.
    if (!IsFusilliCompatibleOp(node.GetOperatorType())) {
      all_fusilli_compatible_ = false;
    }

    // Look up input TensorAttributes from symbol table
    std::vector<Ort::ConstValueInfo> node_inputs = node.GetInputs();
    std::vector<TensorAttrPtr> input_attrs;
    input_attrs.reserve(node_inputs.size());

    for (const auto& input : node_inputs) {
      std::string name = input.GetName();
      auto it = symbol_table_.find(name);
      if (it == symbol_table_.end()) {
        return Status::Failure("Input not found in symbol table: " + name);
      }
      input_attrs.push_back(it->second);
    }

    // Add the node to hipDNN graph
    std::vector<TensorAttrPtr> output_attrs;
    auto status =
        AddNode(*graph_, node, input_attrs, output_attrs, next_uid_);
    if (status.failed()) return status;

    // Set UID, name on output TensorAttributes and add to symbol table.
    // Filter out empty-named optional outputs (ONNX convention for
    // "not requested").  Only active (non-empty-named) outputs are matched
    // against the attrs returned by the node builder.
    std::vector<Ort::ConstValueInfo> node_outputs = node.GetOutputs();
    std::vector<Ort::ConstValueInfo> active_outputs;
    active_outputs.reserve(node_outputs.size());
    for (const auto& out : node_outputs) {
      if (!out.GetName().empty()) {
        active_outputs.push_back(out);
      }
    }

    if (output_attrs.size() != active_outputs.size()) {
      return Status::Failure("Output count mismatch for node " + node.GetName() +
                             ": expected " + std::to_string(active_outputs.size()) +
                             ", got " + std::to_string(output_attrs.size()));
    }

    for (size_t i = 0; i < output_attrs.size(); ++i) {
      std::string name = active_outputs[i].GetName();

      // Get output data type
      auto dtype = ToHipDNNDataType(GetTensorElementType(active_outputs[i]));
      if (!dtype.has_value()) {
        return Status::Failure("Unsupported data type for output: " + name);
      }

      output_attrs[i]->set_uid(next_uid_++).set_name(name).set_data_type(dtype.value());

      // If the per-op node builder already set dim/stride (e.g. AddConvNode
      // for NHWC), keep them.  Otherwise read shape from the ORT graph and
      // compute row-major strides.
      if (output_attrs[i]->get_dim().empty()) {
        auto shape = GetTensorShape(active_outputs[i]);
        if (!shape.has_value()) {
          return Status::Failure("Output must have static shape: " + name);
        }
        output_attrs[i]->set_dim(shape.value());
        output_attrs[i]->set_stride(ComputeStrides(shape.value()));
      }

      symbol_table_[name] = output_attrs[i];
    }
  }

  // Mark graph outputs as non-virtual and store their UIDs
  output_uids_.reserve(graph_outputs.size());
  for (const auto& output : graph_outputs) {
    std::string name = output.GetName();
    auto it = symbol_table_.find(name);
    if (it == symbol_table_.end()) {
      return Status::Failure("Graph output not found in symbol table: " + name);
    }
    it->second->set_is_virtual(false);
    output_uids_.push_back(it->second->get_uid());
  }

  return Status::Success();
}

#ifdef HIPDNN_EP_HAS_TORCH_MLIR
Status HipDNNGraphImpl::Build(mlir::Region& region) {
  using namespace hipdnn_frontend::graph;

  if (region.empty()) {
    return Status::Failure("Empty region in hipdnn.graph");
  }

  mlir::Block& block = region.front();

  auto* terminator = block.getTerminator();
  if (!terminator) {
    return Status::Failure("Region has no terminator");
  }

  output_shapes_.reserve(terminator->getNumOperands());
  for (mlir::Value output : terminator->getOperands()) {
    auto info = GetTensorInfo(output.getLoc(), output.getType());
    if (mlir::failed(info)) {
      return Status::Failure("Failed to get tensor info for output");
    }
    output_shapes_.push_back(info->shape);
  }

  graph_ = std::make_unique<Graph>();

  llvm::DenseMap<mlir::Value, TensorAttrPtr> value_map;

  // Only process tensor-type block arguments (skip non-tensor args like
  // !torch.none, !torch.list<int>, etc.)
  for (auto [idx, arg] : llvm::enumerate(block.getArguments())) {
    if (!mlir::isa<mlir::torch::Torch::ValueTensorType>(arg.getType())) {
      continue;
    }
    std::string name = "input_" + std::to_string(idx);
    auto attr =
        CreateTensorAttrFromMLIR(arg.getLoc(), arg.getType(), next_uid_++, name);
    if (mlir::failed(attr)) {
      return Status::Failure("Failed to create tensor attr for input " +
                             std::to_string(idx));
    }
    (*attr)->set_is_virtual(false);
    value_map[arg] = *attr;
    input_uids_.push_back((*attr)->get_uid());
  }

  for (mlir::Operation& op : block.without_terminator()) {
    if (op.getNumResults() == 0) {
      continue;
    }

    bool hasTensorResult = false;
    for (mlir::Value result : op.getResults()) {
      if (mlir::isa<mlir::torch::Torch::ValueTensorType>(result.getType())) {
        hasTensorResult = true;
        break;
      }
    }
    if (!hasTensorResult) {
      continue;
    }

    std::vector<TensorAttrPtr> input_attrs;
    for (mlir::Value operand : op.getOperands()) {
      auto it = value_map.find(operand);
      if (it != value_map.end()) {
        input_attrs.push_back(it->second);
      }
    }

    // Track whether all ops are Fusilli-compatible.
    if (!IsFusilliCompatibleMLIROp(op.getName().getStringRef())) {
      all_fusilli_compatible_ = false;
    }

    std::vector<TensorAttrPtr> output_attrs;
    auto status = AddNodeFromMLIR(*graph_, &op, input_attrs, output_attrs,
                                  next_uid_);
    if (status.failed()) return status;

    size_t tensor_result_idx = 0;
    for (mlir::Value result : op.getResults()) {
      if (!mlir::isa<mlir::torch::Torch::ValueTensorType>(result.getType())) {
        continue;
      }
      if (tensor_result_idx >= output_attrs.size()) {
        break;
      }

      TensorAttrPtr& attr = output_attrs[tensor_result_idx++];
      auto info = GetTensorInfo(result.getLoc(), result.getType());
      if (mlir::failed(info)) {
        return Status::Failure("Failed to get tensor info for op result");
      }
      attr->set_uid(next_uid_++)
          .set_name("v" + std::to_string(attr->get_uid()))
          .set_data_type(info->dtype)
          .set_dim(info->shape)
          .set_stride(ComputeStrides(info->shape));
      value_map[result] = attr;
    }
  }

  output_uids_.reserve(terminator->getNumOperands());
  for (mlir::Value output : terminator->getOperands()) {
    auto it = value_map.find(output);
    if (it == value_map.end()) {
      return Status::Failure("Output value not found in value map");
    }
    it->second->set_is_virtual(false);
    output_uids_.push_back(it->second->get_uid());
  }

  return Status::Success();
}
#endif  // HIPDNN_EP_HAS_TORCH_MLIR

Status HipDNNGraphImpl::Compile() {
  using hipdnn_frontend::HeuristicMode;

  auto error = graph_->validate();
  if (error.is_bad()) {
    return Status::Failure("hipDNN graph validation failed: " + error.get_message());
  }

  error = graph_->build_operation_graph(handle_);
  if (error.is_bad()) {
    return Status::Failure("hipDNN build_operation_graph failed: " + error.get_message());
  }

  if (all_fusilli_compatible_) {
    graph_->set_preferred_engine_id_ext("FUSILLI_ENGINE");
  }

  error = graph_->create_execution_plans({HeuristicMode::FALLBACK});
  if (error.is_bad()) {
    return Status::Failure("hipDNN create_execution_plans failed: " + error.get_message());
  }

  error = graph_->check_support();
  if (error.is_bad()) {
    return Status::Failure("hipDNN check_support failed: " + error.get_message());
  }

  error = graph_->build_plans();
  if (error.is_bad()) {
    return Status::Failure("hipDNN build_plans failed: " + error.get_message());
  }

  // Get workspace size
  int64_t workspace_size = 0;
  error = graph_->get_workspace_size(workspace_size);
  if (error.is_bad()) {
    return Status::Failure("hipDNN get_workspace_size failed: " + error.get_message());
  }

  if (workspace_size > 0) {
    workspace_.resize(workspace_size);
  }

  return Status::Success();
}

Status HipDNNGraphImpl::Execute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext context(kernel_ctx);

    // Validate input/output counts match what we compiled for.
    // input_uids_ has one entry per ORT graph input (including embedded
    // scalars marked with kEmbeddedScalar).
    if (context.GetInputCount() != input_uids_.size()) {
      return Status::Failure("Input count mismatch: expected " +
                             std::to_string(input_uids_.size()) + ", got " +
                             std::to_string(context.GetInputCount()));
    }
    if (context.GetOutputCount() != output_uids_.size()) {
      return Status::Failure("Output count mismatch: expected " +
                             std::to_string(output_uids_.size()) + ", got " +
                             std::to_string(context.GetOutputCount()));
    }

    // Build variant pack mapping UIDs to data pointers
    std::unordered_map<int64_t, void*> variant_pack;

    // Map graph inputs to their UIDs.
    // Embedded scalars (kEmbeddedScalar) are skipped — their values are
    // baked into the graph via pass-by-value.
    for (size_t i = 0; i < input_uids_.size(); ++i) {
      if (input_uids_[i] == kEmbeddedScalar) continue;
      Ort::ConstValue input = context.GetInput(i);
      variant_pack[input_uids_[i]] = const_cast<void*>(input.GetTensorRawData());
    }

    // Allocate outputs and map to their UIDs
    for (size_t i = 0; i < output_uids_.size(); ++i) {
      Ort::UnownedValue output = context.GetOutput(i, output_shapes_[i]);
      variant_pack[output_uids_[i]] = output.GetTensorMutableRawData();
    }

    // Execute
    void* workspace_ptr = workspace_.empty() ? nullptr : workspace_.data();
    auto error = graph_->execute(handle_, variant_pack, workspace_ptr);
    if (error.is_bad()) {
      return Status::Failure("hipDNN execute failed: " + error.get_message());
    }

  } catch (const Ort::Exception& ex) {
    return Status::Failure(std::string("ORT exception: ") + ex.what());
  }

  return Status::Success();
}

//
// HipDNNGraph public interface
//

HipDNNGraph::HipDNNGraph(hipdnnHandle_t handle)
    : impl_(std::make_unique<HipDNNGraphImpl>(handle)) {}

HipDNNGraph::~HipDNNGraph() = default;

Status HipDNNGraph::Build(
    const std::vector<Ort::ConstValueInfo>& graph_inputs,
    const std::vector<Ort::ConstValueInfo>& graph_outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  return impl_->Build(graph_inputs, graph_outputs, nodes);
}

#ifdef HIPDNN_EP_HAS_TORCH_MLIR
Status HipDNNGraph::Build(mlir::Region& region) {
  return impl_->Build(region);
}
#endif

Status HipDNNGraph::Compile() {
  return impl_->Compile();
}

Status HipDNNGraph::Execute(OrtKernelContext* kernel_ctx) {
  return impl_->Execute(kernel_ctx);
}

}  // namespace hipdnn_ep
