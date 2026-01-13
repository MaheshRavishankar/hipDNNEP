// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/kernel.h"

#include <iostream>

namespace hipdnn_ep {

namespace {

// Helper to check MIOpen status
#define MIOPEN_CHECK(call)                                                  \
  do {                                                                       \
    miopenStatus_t status = (call);                                         \
    if (status != miopenStatusSuccess) {                                    \
      std::cerr << "MIOpen error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }                                                                        \
  } while (0)

#define MIOPEN_RETURN_IF_ERROR(ort_api, call)                               \
  do {                                                                       \
    miopenStatus_t status = (call);                                         \
    if (status != miopenStatusSuccess) {                                    \
      RETURN_ERROR(ort_api, ORT_EP_FAIL, "MIOpen error: " << status);       \
    }                                                                        \
  } while (0)

// Convert ONNX data type to MIOpen data type
miopenDataType_t ToMIOpenDataType(ONNXTensorElementDataType onnx_dtype) {
  switch (onnx_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return miopenFloat;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return miopenHalf;
    default:
      return miopenFloat;
  }
}

}  // namespace

Kernel::Kernel(const OrtApi& ort_api, const OrtLogger& logger)
    : ort_api_(ort_api), logger_(logger) {
  // Create MIOpen handle
  miopenStatus_t status = miopenCreate(&miopen_handle_);
  if (status != miopenStatusSuccess) {
    std::cerr << "Failed to create MIOpen handle: " << status << std::endl;
  }
}

Kernel::~Kernel() {
  // Free workspace
  if (workspace_ != nullptr) {
    hipFree(workspace_);
    workspace_ = nullptr;
  }

  // Destroy descriptors
  if (x_desc_) miopenDestroyTensorDescriptor(x_desc_);
  if (w_desc_) miopenDestroyTensorDescriptor(w_desc_);
  if (y_desc_) miopenDestroyTensorDescriptor(y_desc_);
  if (b_desc_) miopenDestroyTensorDescriptor(b_desc_);
  if (conv_desc_) miopenDestroyConvolutionDescriptor(conv_desc_);

  // Destroy MIOpen handle
  if (miopen_handle_) {
    miopenDestroy(miopen_handle_);
    miopen_handle_ = nullptr;
  }
}

OrtStatus* Kernel::BuildAndCompile(Ort::ConstGraph graph) {
  try {
    std::cerr << "MIOpen Kernel::BuildAndCompile" << std::endl;

    // Get graph inputs and outputs
    std::vector<Ort::ConstValueInfo> graph_inputs = graph.GetInputs();
    std::vector<Ort::ConstValueInfo> graph_outputs = graph.GetOutputs();
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();

    if (nodes.empty()) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Empty graph");
    }

    num_inputs_ = graph_inputs.size();
    num_outputs_ = graph_outputs.size();

    // We expect a single Conv node
    Ort::ConstNode conv_node = nodes[0];
    std::string op_type = conv_node.GetOperatorType();
    if (op_type != "Conv") {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Expected Conv node, got: " << op_type);
    }

    // Get node inputs
    std::vector<Ort::ConstValueInfo> node_inputs = conv_node.GetInputs();
    std::vector<Ort::ConstValueInfo> node_outputs = conv_node.GetOutputs();

    if (node_inputs.size() < 2) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Conv requires at least 2 inputs");
    }

    has_bias_ = node_inputs.size() >= 3;

    // Get input shape (X)
    auto x_shape_opt = GetTensorShape(node_inputs[0]);
    if (!x_shape_opt.has_value()) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Input must have static shape");
    }
    x_shape_ = x_shape_opt.value();

    // Get weight shape (W)
    auto w_shape_opt = GetTensorShape(node_inputs[1]);
    if (!w_shape_opt.has_value()) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Weight must have static shape");
    }
    w_shape_ = w_shape_opt.value();

    // Get bias shape (B) if present
    if (has_bias_) {
      auto b_shape_opt = GetTensorShape(node_inputs[2]);
      if (!b_shape_opt.has_value()) {
        RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Bias must have static shape");
      }
      b_shape_ = b_shape_opt.value();
    }

    // Get output shape (Y)
    auto y_shape_opt = GetTensorShape(node_outputs[0]);
    if (!y_shape_opt.has_value()) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Output must have static shape");
    }
    y_shape_ = y_shape_opt.value();
    output_shapes_.push_back(y_shape_);

    // Get data type
    data_type_ = ToMIOpenDataType(GetTensorElementType(node_inputs[0]));

    // Get convolution attributes
    std::vector<int64_t> pads = GetIntsAttrOrDefault(conv_node, "pads", {0, 0, 0, 0});
    std::vector<int64_t> strides = GetIntsAttrOrDefault(conv_node, "strides", {1, 1});
    std::vector<int64_t> dilations = GetIntsAttrOrDefault(conv_node, "dilations", {1, 1});

    // Normalize pads
    if (pads.size() == 2) {
      pads = {pads[0], pads[1], pads[0], pads[1]};
    }

    std::cerr << "Input shape: [" << x_shape_[0] << ", " << x_shape_[1] << ", " 
              << x_shape_[2] << ", " << x_shape_[3] << "]" << std::endl;
    std::cerr << "Weight shape: [" << w_shape_[0] << ", " << w_shape_[1] << ", " 
              << w_shape_[2] << ", " << w_shape_[3] << "]" << std::endl;
    std::cerr << "Output shape: [" << y_shape_[0] << ", " << y_shape_[1] << ", " 
              << y_shape_[2] << ", " << y_shape_[3] << "]" << std::endl;
    std::cerr << "Has bias: " << has_bias_ << std::endl;

    // Create tensor descriptors
    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenCreateTensorDescriptor(&x_desc_));
    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenCreateTensorDescriptor(&w_desc_));
    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenCreateTensorDescriptor(&y_desc_));

    // Set tensor descriptors (NCHW format)
    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenSet4dTensorDescriptor(
        x_desc_, data_type_,
        static_cast<int>(x_shape_[0]), static_cast<int>(x_shape_[1]),
        static_cast<int>(x_shape_[2]), static_cast<int>(x_shape_[3])));

    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenSet4dTensorDescriptor(
        w_desc_, data_type_,
        static_cast<int>(w_shape_[0]), static_cast<int>(w_shape_[1]),
        static_cast<int>(w_shape_[2]), static_cast<int>(w_shape_[3])));

    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenSet4dTensorDescriptor(
        y_desc_, data_type_,
        static_cast<int>(y_shape_[0]), static_cast<int>(y_shape_[1]),
        static_cast<int>(y_shape_[2]), static_cast<int>(y_shape_[3])));

    // Create and set bias descriptor if needed
    if (has_bias_) {
      MIOPEN_RETURN_IF_ERROR(ort_api_, miopenCreateTensorDescriptor(&b_desc_));
      // Bias is 1D [C], we set it as [1, C, 1, 1] for broadcasting
      // Use explicit strides for proper broadcasting behavior
      int b_dims[4] = {1, static_cast<int>(b_shape_[0]), 1, 1};
      int b_strides[4] = {static_cast<int>(b_shape_[0]), 1, 1, 1};
      MIOPEN_RETURN_IF_ERROR(ort_api_, miopenSetTensorDescriptor(
          b_desc_, data_type_, 4, b_dims, b_strides));
    }

    // Create convolution descriptor
    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenCreateConvolutionDescriptor(&conv_desc_));
    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenInitConvolutionDescriptor(
        conv_desc_,
        miopenConvolution,  // mode
        static_cast<int>(pads[0]),      // pad_h
        static_cast<int>(pads[1]),      // pad_w
        static_cast<int>(strides[0]),   // stride_h
        static_cast<int>(strides[1]),   // stride_w
        static_cast<int>(dilations[0]), // dilation_h
        static_cast<int>(dilations[1])  // dilation_w
    ));

    // Get workspace size first
    MIOPEN_RETURN_IF_ERROR(ort_api_, miopenConvolutionForwardGetWorkSpaceSize(
        miopen_handle_,
        w_desc_,
        x_desc_,
        conv_desc_,
        y_desc_,
        &workspace_size_));

    std::cerr << "Workspace size: " << workspace_size_ << std::endl;

    // Allocate workspace
    if (workspace_size_ > 0) {
      hipError_t hip_err = hipMalloc(&workspace_, workspace_size_);
      if (hip_err != hipSuccess) {
        RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Failed to allocate workspace: " << hipGetErrorString(hip_err));
      }
    }

    // Allocate temporary GPU buffers for finding algorithm
    void* x_tmp = nullptr;
    void* w_tmp = nullptr;
    void* y_tmp = nullptr;
    
    size_t x_size = x_shape_[0] * x_shape_[1] * x_shape_[2] * x_shape_[3] * sizeof(float);
    size_t w_size = w_shape_[0] * w_shape_[1] * w_shape_[2] * w_shape_[3] * sizeof(float);
    size_t y_size = y_shape_[0] * y_shape_[1] * y_shape_[2] * y_shape_[3] * sizeof(float);
    
    if (data_type_ == miopenHalf) {
      x_size /= 2;
      w_size /= 2;
      y_size /= 2;
    }

    hipError_t hip_err = hipMalloc(&x_tmp, x_size);
    if (hip_err != hipSuccess) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Failed to allocate x_tmp: " << hipGetErrorString(hip_err));
    }
    
    hip_err = hipMalloc(&w_tmp, w_size);
    if (hip_err != hipSuccess) {
      hipFree(x_tmp);
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Failed to allocate w_tmp: " << hipGetErrorString(hip_err));
    }
    
    hip_err = hipMalloc(&y_tmp, y_size);
    if (hip_err != hipSuccess) {
      hipFree(x_tmp);
      hipFree(w_tmp);
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Failed to allocate y_tmp: " << hipGetErrorString(hip_err));
    }

    // Find the best convolution algorithm (required by MIOpen)
    const int request_algo_count = 4;
    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf_results[request_algo_count];

    std::cerr << "Finding convolution algorithm..." << std::endl;
    miopenStatus_t find_status = miopenFindConvolutionForwardAlgorithm(
        miopen_handle_,
        x_desc_,
        x_tmp,
        w_desc_,
        w_tmp,
        conv_desc_,
        y_desc_,
        y_tmp,
        request_algo_count,
        &returned_algo_count,
        perf_results,
        workspace_,
        workspace_size_,
        false  // exhaustiveSearch
    );

    // Free temporary buffers
    hipFree(x_tmp);
    hipFree(w_tmp);
    hipFree(y_tmp);

    if (find_status != miopenStatusSuccess) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "miopenFindConvolutionForwardAlgorithm failed: " << find_status);
    }

    if (returned_algo_count == 0) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "No convolution algorithm found");
    }

    // Use the best algorithm found
    conv_algo_ = perf_results[0].fwd_algo;
    std::cerr << "Selected algorithm: " << conv_algo_ << ", time: " << perf_results[0].time << " ms" << std::endl;

    std::cerr << "MIOpen Kernel::BuildAndCompile complete" << std::endl;

  } catch (const std::exception& ex) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Exception building MIOpen kernel: " << ex.what());
  }

  return nullptr;
}

OrtStatus* Kernel::Execute(OrtKernelContext* kernel_ctx) {
  try {
    std::cerr << "MIOpen Kernel::Execute" << std::endl;

    Ort::KernelContext context(kernel_ctx);

    // Validate input/output counts
    if (context.GetInputCount() < 2) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Expected at least 2 inputs, got " << context.GetInputCount());
    }

    // Get input pointers
    Ort::ConstValue x_tensor = context.GetInput(0);
    Ort::ConstValue w_tensor = context.GetInput(1);
    const void* x_ptr = x_tensor.GetTensorRawData();
    const void* w_ptr = w_tensor.GetTensorRawData();

    // Allocate output
    Ort::UnownedValue y_tensor = context.GetOutput(0, output_shapes_[0]);
    void* y_ptr = y_tensor.GetTensorMutableRawData();

    // Scaling factors
    float alpha = 1.0f;
    float beta = 0.0f;

    miopenStatus_t status;

    // Execute convolution: y = conv(x, w)
    // Note: MIOpen only supports alpha=1 and beta=0 for 2D convolutions
    std::cerr << "Executing miopenConvolutionForward..." << std::endl;
    
    status = miopenConvolutionForward(
        miopen_handle_,
        &alpha,
        x_desc_,
        x_ptr,
        w_desc_,
        w_ptr,
        conv_desc_,
        conv_algo_,
        &beta,
        y_desc_,
        y_ptr,
        workspace_,
        workspace_size_);

    if (status != miopenStatusSuccess) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "miopenConvolutionForward failed: " << status);
    }

    // Add bias if present: y = y + bias
    if (has_bias_ && context.GetInputCount() >= 3) {
      std::cerr << "Adding bias with miopenOpTensor..." << std::endl;
      
      Ort::ConstValue b_tensor = context.GetInput(2);
      const void* b_ptr = b_tensor.GetTensorRawData();

      // y = 1*y + 1*bias + 0*y = y + bias
      float alpha1 = 1.0f;
      float alpha2 = 1.0f;
      float beta_op = 0.0f;
      
      status = miopenOpTensor(
          miopen_handle_,
          miopenTensorOpAdd,
          &alpha1,
          y_desc_,
          y_ptr,
          &alpha2,
          b_desc_,
          b_ptr,
          &beta_op,
          y_desc_,
          y_ptr);

      if (status != miopenStatusSuccess) {
        RETURN_ERROR(ort_api_, ORT_EP_FAIL, "miopenOpTensor (bias add) failed: " << status);
      }
    }

    std::cerr << "MIOpen Kernel::Execute complete" << std::endl;

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    RETURN_ERROR(ort_api_, ORT_EP_FAIL, "Exception in MIOpen Kernel::Execute: " << ex.what());
  }

  return nullptr;
}

}  // namespace hipdnn_ep
