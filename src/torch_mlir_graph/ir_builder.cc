// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/torch_mlir_graph/ir_builder.h"
#include "hipdnn_ep/utils/ep_utils.h"

#ifdef HIPDNN_EP_HAS_TORCH_MLIR

#include "hipdnn_ep/torch_mlir_graph/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace hipdnn_ep {

namespace {

// Get input names for a node
static std::vector<std::string> GetNodeInputNames(Ort::ConstNode node) {
  std::vector<std::string> names;
  for (const auto& input : node.GetInputs()) {
    names.push_back(input.GetName());
  }
  return names;
}

// Get output names for a node
static std::vector<std::string> GetNodeOutputNames(Ort::ConstNode node) {
  std::vector<std::string> names;
  for (const auto& output : node.GetOutputs()) {
    names.push_back(output.GetName());
  }
  return names;
}

// Convert ONNX element type to MLIR type
static mlir::FailureOr<mlir::Type> ConvertElementType(
    mlir::MLIRContext* ctx, ONNXTensorElementDataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return mlir::Float32Type::get(ctx);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return mlir::Float16Type::get(ctx);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Signed);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
    default:
      return mlir::emitError(mlir::UnknownLoc::get(ctx))
             << "unsupported ONNX element type: " << static_cast<int>(dtype);
  }
}

// Convert ONNX tensor type to Torch MLIR tensor type
static mlir::FailureOr<mlir::Type> ConvertTensorType(
    mlir::MLIRContext* ctx, ONNXTensorElementDataType dtype,
    const std::vector<int64_t>& shape) {
  auto elem_type = ConvertElementType(ctx, dtype);
  if (mlir::failed(elem_type))
    return mlir::failure();
  return mlir::torch::Torch::ValueTensorType::get(ctx, shape, *elem_type);
}

// Convert ORT attribute to MLIR attribute
static mlir::FailureOr<mlir::Attribute> ConvertAttribute(
    mlir::MLIRContext* ctx, Ort::ConstOpAttr attr) {
  auto int_type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
  auto float_type = mlir::Float32Type::get(ctx);
  std::string attr_name = attr.GetName();

  switch (attr.GetType()) {
    case ORT_OP_ATTR_INT: {
      int64_t value;
      if (!attr.GetValue(value).IsOK())
        return mlir::emitError(mlir::UnknownLoc::get(ctx))
               << "failed to read int attribute '" << attr_name << "'";
      return mlir::IntegerAttr::get(int_type, value);
    }
    case ORT_OP_ATTR_FLOAT: {
      float value;
      if (!attr.GetValue(value).IsOK())
        return mlir::emitError(mlir::UnknownLoc::get(ctx))
               << "failed to read float attribute '" << attr_name << "'";
      return mlir::FloatAttr::get(float_type, value);
    }
    case ORT_OP_ATTR_STRING: {
      std::string value;
      if (!attr.GetValue(value).IsOK())
        return mlir::emitError(mlir::UnknownLoc::get(ctx))
               << "failed to read string attribute '" << attr_name << "'";
      return mlir::StringAttr::get(ctx, value);
    }
    case ORT_OP_ATTR_INTS: {
      std::vector<int64_t> values;
      if (!attr.GetValueArray(values).IsOK())
        return mlir::emitError(mlir::UnknownLoc::get(ctx))
               << "failed to read int array attribute '" << attr_name << "'";
      llvm::SmallVector<mlir::Attribute> elements;
      for (int64_t v : values)
        elements.push_back(mlir::IntegerAttr::get(int_type, v));
      return mlir::ArrayAttr::get(ctx, elements);
    }
    case ORT_OP_ATTR_FLOATS: {
      std::vector<float> values;
      if (!attr.GetValueArray(values).IsOK())
        return mlir::emitError(mlir::UnknownLoc::get(ctx))
               << "failed to read float array attribute '" << attr_name << "'";
      llvm::SmallVector<mlir::Attribute> elements;
      for (float v : values)
        elements.push_back(mlir::FloatAttr::get(float_type, v));
      return mlir::ArrayAttr::get(ctx, elements);
    }
    default:
      return mlir::emitError(mlir::UnknownLoc::get(ctx))
             << "unsupported attribute type for '" << attr_name << "': "
             << attr.GetType();
  }
}

// Build a generic ONNX operation using torch.operator "onnx.*"
static mlir::FailureOr<llvm::SmallVector<mlir::Value>> BuildOnnxOp(
    mlir::MLIRContext* ctx, mlir::OpBuilder* builder, Ort::ConstNode node,
    llvm::ArrayRef<mlir::Value> inputs, llvm::ArrayRef<mlir::Type> result_types) {
  auto loc = mlir::UnknownLoc::get(ctx);
  std::string op_name = "onnx." + node.GetOperatorType();

  // Create the torch.operator op (0 regions)
  auto op = mlir::torch::Torch::OperatorOp::create(
      *builder, loc, result_types, builder->getStringAttr(op_name), inputs, 0);

  // Convert and set all attributes with "torch.onnx." prefix
  for (const auto& attr : node.GetAttributes()) {
    auto mlir_attr = ConvertAttribute(ctx, attr);
    if (mlir::failed(mlir_attr))
      return mlir::failure();
    op->setAttr("torch.onnx." + attr.GetName(), *mlir_attr);
  }

  // Collect results
  llvm::SmallVector<mlir::Value> results;
  for (auto result : op->getResults()) {
    results.push_back(result);
  }

  return results;
}

}  // namespace

//
// IRBuilderImpl - MLIR-specific implementation details
//

struct IRBuilderImpl {
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  // Cache of compiled hipDNN graphs, keyed by unique name
  llvm::StringMap<std::unique_ptr<HipDNNGraph>> compiled_graphs;

  IRBuilderImpl() {
    ctx.loadDialect<mlir::torch::Torch::TorchDialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
  }

  bool BuildModule(const std::vector<Ort::ConstValueInfo>& inputs,
                   const std::vector<Ort::ConstValueInfo>& outputs,
                   const std::vector<Ort::ConstNode>& nodes);

  std::string PrintModule() const;

  bool RunOffloadPipeline(hipdnnHandle_t handle);

  HipDNNGraph* GetCompiledGraph(const std::string& name);

  size_t GetCompiledGraphCount() const { return compiled_graphs.size(); }
};

bool IRBuilderImpl::BuildModule(
    const std::vector<Ort::ConstValueInfo>& inputs,
    const std::vector<Ort::ConstValueInfo>& outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  // Create module
  module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  mlir::OpBuilder builder(module->getBodyRegion());

  // Build function argument types
  llvm::SmallVector<mlir::Type> arg_types;
  llvm::StringMap<mlir::Value> value_map;

  for (const auto& input : inputs) {
    auto shape = GetTensorShape(input);
    if (!shape.has_value()) {
      return false;
    }

    auto dtype = GetTensorElementType(input);
    auto tensor_type = ConvertTensorType(&ctx, dtype, shape.value());
    if (mlir::failed(tensor_type)) {
      return false;
    }
    arg_types.push_back(*tensor_type);
  }

  // Build function result types
  llvm::SmallVector<mlir::Type> result_types;
  for (const auto& output : outputs) {
    auto shape = GetTensorShape(output);
    if (!shape.has_value()) {
      return false;
    }

    auto dtype = GetTensorElementType(output);
    auto tensor_type = ConvertTensorType(&ctx, dtype, shape.value());
    if (mlir::failed(tensor_type)) {
      return false;
    }
    result_types.push_back(*tensor_type);
  }

  // Create function type and function op
  auto func_type = mlir::FunctionType::get(&ctx, arg_types, result_types);
  auto func = mlir::func::FuncOp::create(
      builder, mlir::UnknownLoc::get(&ctx), "main", func_type);

  // Add ONNX opset version attribute (required by TorchOnnxToTorch pass)
  // Note: Must be a signed integer type for torch-mlir to accept it
  func->setAttr("torch.onnx_meta.opset_version",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(&ctx, 64, mlir::IntegerType::Signed),
                    14));

  // Create entry block with arguments
  mlir::Block* entry_block = func.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  // Map input names to block arguments
  for (size_t i = 0; i < inputs.size(); ++i) {
    value_map[inputs[i].GetName()] = entry_block->getArgument(i);
  }

  // Process nodes in topological order
  for (const auto& node : nodes) {
    // Gather input values
    auto input_names = GetNodeInputNames(node);
    llvm::SmallVector<mlir::Value> input_values;

    for (const auto& name : input_names) {
      auto it = value_map.find(name);
      if (it == value_map.end()) {
        mlir::emitError(mlir::UnknownLoc::get(&ctx))
            << "input value not found: " << name;
        return false;
      }
      input_values.push_back(it->second);
    }

    // Get output types for this node
    auto output_infos = node.GetOutputs();
    llvm::SmallVector<mlir::Type> node_result_types;
    for (const auto& output_info : output_infos) {
      auto shape = GetTensorShape(output_info);
      if (!shape.has_value()) {
        mlir::emitError(mlir::UnknownLoc::get(&ctx))
            << "failed to get shape for output: " << output_info.GetName();
        return false;
      }
      auto dtype = GetTensorElementType(output_info);
      auto tensor_type = ConvertTensorType(&ctx, dtype, *shape);
      if (mlir::failed(tensor_type)) {
        return false;
      }
      node_result_types.push_back(*tensor_type);
    }

    // Build the ONNX operation
    auto results = BuildOnnxOp(&ctx, &builder, node,
                               input_values, node_result_types);
    if (mlir::failed(results)) {
      return false;
    }

    // Map output names to results
    auto output_names = GetNodeOutputNames(node);
    for (size_t i = 0; i < results->size() && i < output_names.size(); ++i) {
      value_map[output_names[i]] = (*results)[i];
    }
  }

  // Build return statement
  llvm::SmallVector<mlir::Value> return_values;
  for (const auto& output : outputs) {
    auto it = value_map.find(output.GetName());
    if (it == value_map.end()) {
      mlir::emitError(mlir::UnknownLoc::get(&ctx))
          << "output value not found: " << output.GetName();
      return false;
    }
    return_values.push_back(it->second);
  }

  mlir::func::ReturnOp::create(builder, mlir::UnknownLoc::get(&ctx),
                               return_values);

  return true;
}

std::string IRBuilderImpl::PrintModule() const {
  if (!module) {
    return "";
  }
  std::string result;
  llvm::raw_string_ostream os(result);
  module.get().print(os);
  return result;
}

bool IRBuilderImpl::RunOffloadPipeline(hipdnnHandle_t handle) {
  if (!module) {
    return false;
  }

  // Create output map for compiled graphs
  auto output_graphs =
      std::make_shared<llvm::StringMap<std::unique_ptr<HipDNNGraph>>>();

  mlir::PassManager pm(module->getContext());

  // Step 1: Convert onnx.* ops to aten.* ops
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::torch::onnx_c::createTorchOnnxToTorchPass());

  // Step 2: Apply hipDNN offload pass
  pm.addNestedPass<mlir::func::FuncOp>(createHipDNNOffloadPass());

  // Step 3: Compile hipDNN graphs and transform to executables
  pm.addPass(createHipDNNGraphToExecutablePass(handle, output_graphs));

  if (mlir::failed(pm.run(*module))) {
    return false;
  }

  // Take ownership of all compiled graphs
  for (auto& entry : *output_graphs) {
    compiled_graphs[entry.first()] = std::move(entry.second);
  }

  return true;
}

HipDNNGraph* IRBuilderImpl::GetCompiledGraph(const std::string& name) {
  auto it = compiled_graphs.find(name);
  if (it == compiled_graphs.end()) {
    return nullptr;
  }
  return it->second.get();
}

//
// IRBuilder - public interface delegates to Impl
//

IRBuilder::IRBuilder() : impl_(std::make_unique<IRBuilderImpl>()) {}

IRBuilder::~IRBuilder() = default;

bool IRBuilder::BuildModule(
    const std::vector<Ort::ConstValueInfo>& inputs,
    const std::vector<Ort::ConstValueInfo>& outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  return impl_->BuildModule(inputs, outputs, nodes);
}

std::string IRBuilder::PrintModule() const {
  return impl_->PrintModule();
}

bool IRBuilder::RunOffloadPipeline(hipdnnHandle_t handle) {
  return impl_->RunOffloadPipeline(handle);
}

HipDNNGraph* IRBuilder::GetCompiledGraph(const std::string& name) {
  return impl_->GetCompiledGraph(name);
}

size_t IRBuilder::GetCompiledGraphCount() const {
  return impl_->GetCompiledGraphCount();
}

}  // namespace hipdnn_ep

#else  // !HIPDNN_EP_HAS_TORCH_MLIR

namespace hipdnn_ep {

// Define empty impl so unique_ptr can be destructed
struct IRBuilderImpl {};

IRBuilder::IRBuilder() = default;
IRBuilder::~IRBuilder() = default;

bool IRBuilder::BuildModule(const std::vector<Ort::ConstValueInfo>& /*inputs*/,
                            const std::vector<Ort::ConstValueInfo>& /*outputs*/,
                            const std::vector<Ort::ConstNode>& /*nodes*/) {
  return false;
}

std::string IRBuilder::PrintModule() const { return ""; }

bool IRBuilder::RunOffloadPipeline(hipdnnHandle_t /*handle*/) { return false; }

HipDNNGraph* IRBuilder::GetCompiledGraph(const std::string& /*name*/) {
  return nullptr;
}

size_t IRBuilder::GetCompiledGraphCount() const { return 0; }

}  // namespace hipdnn_ep

#endif  // HIPDNN_EP_HAS_TORCH_MLIR
