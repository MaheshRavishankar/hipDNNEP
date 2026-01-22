// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/ep.h"
#include "hipdnn_ep/ep_factory.h"
#include "hipdnn_ep/kernel.h"
#include "hipdnn_ep/node_compute_info.h"

#include "hipdnn_ep/blas_graph.h"

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

    // Check C shape if present (must match output shape, no broadcasting)
    if (inputs.size() == 3) {
      auto c_shape = GetTensorShape(inputs[2]);
      if (!c_shape.has_value() || c_shape->size() != 2) {
        return false;
      }

      int64_t m = trans_a ? (*a_shape)[1] : (*a_shape)[0];
      int64_t n = trans_b ? (*b_shape)[0] : (*b_shape)[1];

      if ((*c_shape)[0] != m || (*c_shape)[1] != n) {
        return false;  // C must match output shape (no broadcasting)
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

      // Create kernel and build/compile the graph
      KernelConfig kernel_config;
      kernel_config.setHipDNNHandle(ep->hipdnn_handle_)
          .setHipBlasLtHandle(ep->hipblaslt_handle_);
      if (ep->config_.use_torch_mlir) {
        kernel_config.setUseTorchMlir(ep->config_.dump_torch_mlir);
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
