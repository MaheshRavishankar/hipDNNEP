// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/kernel.h"

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
OrtStatus* CreateTensorAttr(
    const OrtApi& ort_api,
    Ort::ConstValueInfo value_info,
    int64_t uid,
    TensorAttrPtr& out_attr) {
  using hipdnn_frontend::graph::TensorAttributes;

  std::string name = value_info.GetName();

  auto shape = GetTensorShape(value_info);
  if (!shape.has_value()) {
    RETURN_ERROR(ort_api, ORT_EP_FAIL, "Value must have static shape: " << name);
  }

  auto dtype = ToHipDNNDataType(GetTensorElementType(value_info));
  if (!dtype.has_value()) {
    RETURN_ERROR(ort_api, ORT_EP_FAIL, "Unsupported data type for value: " << name);
  }

  out_attr = std::make_shared<TensorAttributes>();
  out_attr->set_uid(uid)
          .set_name(name)
          .set_data_type(dtype.value())
          .set_dim(shape.value())
          .set_stride(ComputeStrides(shape.value()));

  return nullptr;
}

// Add Conv operation to hipDNN graph
// Takes input tensor attributes (X, W), returns output tensor attribute (Y)
OrtStatus* AddConvNode(
    const OrtApi& ort_api,
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    TensorAttrPtr& output_attr) {
  using namespace hipdnn_frontend::graph;
  using hipdnn_frontend::ConvolutionMode;

  if (input_attrs.size() < 2) {
    RETURN_ERROR(ort_api, ORT_EP_FAIL, "Conv requires at least 2 input tensor attributes");
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
    RETURN_ERROR(ort_api, ORT_EP_FAIL, "Conv pads must have 2 or 4 elements");
  }

  // Determine compute data type from input data types
  auto compute_dtype = GetComputeDataType(x_attr->get_data_type(), w_attr->get_data_type());
  if (!compute_dtype.has_value()) {
    RETURN_ERROR(ort_api, ORT_EP_FAIL, "Unsupported data type combination for Conv compute");
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

  return nullptr;
}

// Dispatch to appropriate Add*Node based on op_type
// Takes input tensor attributes, returns output tensor attributes
OrtStatus* AddNode(
    const OrtApi& ort_api,
    hipdnn_frontend::graph::Graph& graph,
    Ort::ConstNode node,
    const std::vector<TensorAttrPtr>& input_attrs,
    std::vector<TensorAttrPtr>& output_attrs) {
  std::string op_type = node.GetOperatorType();

  if (op_type == "Conv") {
    TensorAttrPtr y_attr;
    RETURN_IF_ERROR(AddConvNode(ort_api, graph, node, input_attrs, y_attr));
    output_attrs.push_back(y_attr);
    return nullptr;
  }

  RETURN_ERROR(ort_api, ORT_EP_FAIL, "Unsupported op type: " << op_type);
}

}  // namespace

//
// Kernel implementation
//

Kernel::Kernel(const OrtApi& ort_api, const OrtLogger& logger, hipdnnHandle_t handle)
    : ort_api_(ort_api), logger_(logger), handle_(handle) {
}

Kernel::~Kernel() = default;

OrtStatus* Kernel::BuildAndCompile(Ort::ConstGraph graph) {
  try {
    using namespace hipdnn_frontend::graph;

    graph_ = std::make_unique<Graph>();

    // Extract graph input/output info
    std::vector<Ort::ConstValueInfo> graph_inputs = graph.GetInputs();
    std::vector<Ort::ConstValueInfo> graph_outputs = graph.GetOutputs();

    // Create TensorAttributes for all graph inputs and add to symbol table
    input_uids_.reserve(graph_inputs.size());
    for (const auto& input : graph_inputs) {
      TensorAttrPtr attr;
      RETURN_IF_ERROR(CreateTensorAttr(ort_api_, input, next_uid_++, attr));
      attr->set_is_virtual(false);
      symbol_table_[input.GetName()] = attr;
      input_uids_.push_back(attr->get_uid());
    }

    // Process each node in the graph
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();
    for (const auto& node : nodes) {
      // Look up input TensorAttributes from symbol table
      std::vector<Ort::ConstValueInfo> node_inputs = node.GetInputs();
      std::vector<TensorAttrPtr> input_attrs;
      input_attrs.reserve(node_inputs.size());

      for (const auto& input : node_inputs) {
        std::string name = input.GetName();
        auto it = symbol_table_.find(name);
        if (it == symbol_table_.end()) {
          RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Input not found in symbol table: " << name);
        }
        input_attrs.push_back(it->second);
      }

      // Add the node to hipDNN graph
      std::vector<TensorAttrPtr> output_attrs;
      RETURN_IF_ERROR(AddNode(ort_api_, *graph_, node, input_attrs, output_attrs));

      // Set UID, name on output TensorAttributes and add to symbol table
      std::vector<Ort::ConstValueInfo> node_outputs = node.GetOutputs();
      if (output_attrs.size() != node_outputs.size()) {
        RETURN_ERROR(ort_api_, ORT_EP_FAIL,
            "Output count mismatch for node " << node.GetName() <<
            ": expected " << node_outputs.size() << ", got " << output_attrs.size());
      }

      for (size_t i = 0; i < output_attrs.size(); ++i) {
        std::string name = node_outputs[i].GetName();

        // Get output data type
        auto dtype = ToHipDNNDataType(GetTensorElementType(node_outputs[i]));
        if (!dtype.has_value()) {
          RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Unsupported data type for output: " << name);
        }

        // Get output shape for strides
        auto shape = GetTensorShape(node_outputs[i]);
        if (!shape.has_value()) {
          RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Output must have static shape: " << name);
        }

        output_attrs[i]->set_uid(next_uid_++)
                       .set_name(name)
                       .set_data_type(dtype.value())
                       .set_dim(shape.value())
                       .set_stride(ComputeStrides(shape.value()));
        symbol_table_[name] = output_attrs[i];
      }
    }

    // Mark graph outputs as non-virtual and store their UIDs
    output_uids_.reserve(graph_outputs.size());
    output_shapes_.reserve(graph_outputs.size());
    for (const auto& output : graph_outputs) {
      std::string name = output.GetName();
      auto it = symbol_table_.find(name);
      if (it == symbol_table_.end()) {
        RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Graph output not found in symbol table: " << name);
      }
      it->second->set_is_virtual(false);
      output_uids_.push_back(it->second->get_uid());

      auto shape = GetTensorShape(output);
      if (!shape.has_value()) {
        RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Graph output must have static shape: " << name);
      }
      output_shapes_.push_back(shape.value());
    }

    // Compile the graph
    RETURN_IF_ERROR(CompileGraph());

  } catch (const std::exception& ex) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Exception building hipDNN graph: " << ex.what());
  }

  return nullptr;
}

OrtStatus* Kernel::CompileGraph() {
  using hipdnn_frontend::HeuristicMode;

  auto error = graph_->validate();
  if (error.is_bad()) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "hipDNN graph validation failed: " << error.get_message());
  }

  error = graph_->build_operation_graph(handle_);
  if (error.is_bad()) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "hipDNN build_operation_graph failed: " << error.get_message());
  }

  error = graph_->create_execution_plans({HeuristicMode::FALLBACK});
  if (error.is_bad()) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "hipDNN create_execution_plans failed: " << error.get_message());
  }

  error = graph_->check_support();
  if (error.is_bad()) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "hipDNN check_support failed: " << error.get_message());
  }

  error = graph_->build_plans();
  if (error.is_bad()) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "hipDNN build_plans failed: " << error.get_message());
  }

  // Get workspace size
  int64_t workspace_size = 0;
  error = graph_->get_workspace_size(workspace_size);
  if (error.is_bad()) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "hipDNN get_workspace_size failed: " << error.get_message());
  }

  if (workspace_size > 0) {
    workspace_.resize(workspace_size);
  }

  return nullptr;
}

OrtStatus* Kernel::Execute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext context(kernel_ctx);

    // Validate input/output counts match what we compiled for
    if (context.GetInputCount() != input_uids_.size()) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL,
          "Input count mismatch: expected " << input_uids_.size() << ", got " << context.GetInputCount());
    }
    if (context.GetOutputCount() != output_uids_.size()) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL,
          "Output count mismatch: expected " << output_uids_.size() << ", got " << context.GetOutputCount());
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
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "hipDNN execute failed: " << error.get_message());
    }

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Exception in Kernel::Execute: " << ex.what());
  }

  return nullptr;
}

}  // namespace hipdnn_ep
