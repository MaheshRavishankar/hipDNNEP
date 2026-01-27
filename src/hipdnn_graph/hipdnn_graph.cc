// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/hipdnn_graph/hipdnn_graph.h"
#include "hipdnn_ep/utils/ep_utils.h"

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>

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

// Helper function to compute strides from shape (NCHW layout)
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
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

// Determine compute data type based on input data types
// For float types with precision <= float32, compute in float32
std::optional<hipdnn_frontend::DataType> GetComputeDataType(
    hipdnn_frontend::DataType x_dtype,
    hipdnn_frontend::DataType w_dtype) {
  using hipdnn_frontend::DataType;

  // Both must be float types (FLOAT or HALF)
  bool x_is_float = (x_dtype == DataType::FLOAT || x_dtype == DataType::HALF);
  bool w_is_float = (w_dtype == DataType::FLOAT || w_dtype == DataType::HALF);

  if (x_is_float && w_is_float) {
    // Use float32 for compute when inputs are float types with precision <= float32
    return DataType::FLOAT;
  }

  return std::nullopt;
}

using TensorAttrPtr = std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>;

// Create TensorAttributes from a ConstValueInfo
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

// Add Conv operation to hipDNN graph
// Takes input tensor attributes (X, W), returns output tensor attribute (Y)
Status AddConvNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::ConvolutionMode;

  if (input_attrs.size() < 2) {
    return Status::Failure("Conv requires at least 2 input tensor attributes");
  }

  const auto& x_attr = input_attrs[0];
  const auto& w_attr = input_attrs[1];

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

  return Status::Success();
}

// Dispatch to appropriate Add*Node based on op_type
// Takes input tensor attributes, returns output tensor attributes
Status AddNode(
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    std::vector<TensorAttrPtr>& output_attrs) {
  std::string op_type = node.GetOperatorType();

  if (op_type == "Conv") {
    TensorAttrPtr y_attr;
    auto status = AddConvNode(graph, node, input_attrs, y_attr);
    if (status.failed()) return status;
    output_attrs.push_back(y_attr);
    return Status::Success();
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

// Create TensorAttributes from MLIR type
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
                           TensorAttrPtr& output_attr) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::ConvolutionMode;

  if (input_attrs.size() < 2) {
    return Status::Failure("Conv requires at least 2 input tensor attributes");
  }

  const auto& x_attr = input_attrs[0];
  const auto& w_attr = input_attrs[1];

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

  return Status::Success();
}

// Dispatch MLIR op to appropriate Add*Node function
Status AddNodeFromMLIR(hipdnn_frontend::graph::Graph& graph,
                       mlir::Operation* op,
                       const std::vector<TensorAttrPtr>& input_attrs,
                       std::vector<TensorAttrPtr>& output_attrs) {
  llvm::StringRef op_name = op->getName().getStringRef();

  if (op_name == "torch.aten.convolution" || op_name == "torch.aten.conv2d") {
    TensorAttrPtr y_attr;
    auto status = AddConvNodeFromMLIR(graph, op, input_attrs, y_attr);
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

  // Graph input/output info
  std::vector<int64_t> input_uids_;
  std::vector<int64_t> output_uids_;
  std::vector<std::vector<int64_t>> output_shapes_;

  // Symbol table: maps value name to TensorAttributes
  std::unordered_map<std::string, TensorAttrPtr> symbol_table_;

  // UID counter for tensor attributes
  int64_t next_uid_{1};
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

  // Create TensorAttributes for all graph inputs and add to symbol table
  input_uids_.reserve(graph_inputs.size());
  for (const auto& input : graph_inputs) {
    TensorAttrPtr attr;
    auto status = CreateTensorAttr(input, next_uid_++, attr);
    if (status.failed()) return status;
    attr->set_is_virtual(false);
    symbol_table_[input.GetName()] = attr;
    input_uids_.push_back(attr->get_uid());
  }

  // Process each node in the graph
  for (const auto& node : nodes) {
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
    auto status = AddNode(*graph_, node, input_attrs, output_attrs);
    if (status.failed()) return status;

    // Set UID, name on output TensorAttributes and add to symbol table
    std::vector<Ort::ConstValueInfo> node_outputs = node.GetOutputs();
    if (output_attrs.size() != node_outputs.size()) {
      return Status::Failure("Output count mismatch for node " + node.GetName() +
                             ": expected " + std::to_string(node_outputs.size()) +
                             ", got " + std::to_string(output_attrs.size()));
    }

    for (size_t i = 0; i < output_attrs.size(); ++i) {
      std::string name = node_outputs[i].GetName();

      // Get output data type
      auto dtype = ToHipDNNDataType(GetTensorElementType(node_outputs[i]));
      if (!dtype.has_value()) {
        return Status::Failure("Unsupported data type for output: " + name);
      }

      // Get output shape for strides
      auto shape = GetTensorShape(node_outputs[i]);
      if (!shape.has_value()) {
        return Status::Failure("Output must have static shape: " + name);
      }

      output_attrs[i]->set_uid(next_uid_++).set_name(name).set_data_type(dtype.value()).set_dim(shape.value()).set_stride(ComputeStrides(shape.value()));
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

    std::vector<TensorAttrPtr> output_attrs;
    auto status = AddNodeFromMLIR(*graph_, &op, input_attrs, output_attrs);
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

    // Validate input/output counts match what we compiled for
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

    // Map graph inputs to their UIDs
    for (size_t i = 0; i < input_uids_.size(); ++i) {
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
