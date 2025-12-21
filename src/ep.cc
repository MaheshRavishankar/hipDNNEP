// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/ep.h"
#include "hipdnn_ep/ep_factory.h"
#include "hipdnn_ep/kernel.h"
#include "hipdnn_ep/node_compute_info.h"

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

    return true;

  } catch (...) {
    return false;
  }
}

// Check if an op is supported by this EP
static bool IsSupportedOp(Ort::ConstNode node) {
  std::string op_type = node.GetOperatorType();

  if (op_type == "Conv") {
    return IsSupportedConv(node);
  }

  // Add more operations here as we implement them
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

  IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(
      &logger_, ORT_LOGGING_LEVEL_INFO,
      (std::string("HipDNN EP created: ") + factory_.GetName(&factory_)).c_str(),
      EP_FILE, __LINE__, __FUNCTION__));
}

HipDNNEp::~HipDNNEp() {
  kernels_.clear();

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

    for (const auto& node : nodes) {
      if (IsSupportedOp(node)) {
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
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
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

      // Create kernel and build/compile the hipDNN graph
      auto kernel = std::make_unique<Kernel>(ep->ort_api, ep->logger_, ep->hipdnn_handle_);
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
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
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
