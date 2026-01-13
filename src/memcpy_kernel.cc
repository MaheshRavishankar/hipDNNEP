// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/memcpy_kernel.h"
#include "hipdnn_ep/ep_factory.h"

#include <cstring>
#include <iostream>

namespace hipdnn_ep {

MemcpyKernelImpl::MemcpyKernelImpl(HipDNNEpFactory& factory, Direction direction, int device_id)
    : factory_(factory), direction_(direction), device_id_(device_id) {
  ort_version_supported = ORT_API_VERSION;
  flags = 0;
  Compute = ComputeImpl;
  Release = ReleaseImpl;

  // Optional functions - not needed for memcpy
  PrePackWeight = nullptr;
  SetSharedPrePackedWeight = nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL MemcpyKernelImpl::ComputeImpl(
    OrtKernelImpl* this_ptr,
    OrtKernelContext* context) noexcept {
  try {
    auto* impl = static_cast<MemcpyKernelImpl*>(this_ptr);

    Ort::KernelContext ctx(context);

    // Get input and output
    Ort::ConstValue input = ctx.GetInput(0);
    auto input_type_shape = input.GetTensorTypeAndShapeInfo();
    auto shape = input_type_shape.GetShape();

    // Create output with same shape
    std::cerr << "MemcpyKernelImpl::ComputeImpl ctx.GetOutput 0" << std::endl;
    Ort::UnownedValue output = ctx.GetOutput(0, shape);
    std::cerr << "MemcpyKernelImpl::ComputeImpl ctx.GetOutput 1" << std::endl;

    // Get data pointers
    const void* src_data = input.GetTensorRawData();
    void* dst_data = output.GetTensorMutableRawData();

    // Calculate byte size
    size_t element_count = input_type_shape.GetElementCount();
    ONNXTensorElementDataType elem_type = input_type_shape.GetElementType();

    size_t elem_size = 0;
    switch (elem_type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        elem_size = sizeof(float);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        elem_size = 2;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        elem_size = 2;
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        elem_size = sizeof(double);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        elem_size = sizeof(int8_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        elem_size = sizeof(uint8_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        elem_size = sizeof(int16_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        elem_size = sizeof(uint16_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        elem_size = sizeof(int32_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        elem_size = sizeof(uint32_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        elem_size = sizeof(int64_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        elem_size = sizeof(uint64_t);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        elem_size = sizeof(bool);
        break;
      default:
        RETURN_ERROR(impl->factory_.ort_api, ORT_EP_FAIL,
                     "MemcpyKernel: Unsupported tensor element type: " << static_cast<int>(elem_type));
    }

    size_t byte_size = element_count * elem_size;

    // Set the device
    hipError_t err = hipSetDevice(impl->device_id_);
    if (err != hipSuccess) {
      RETURN_ERROR(impl->factory_.ort_api, ORT_EP_FAIL,
                   "MemcpyKernel: Failed to set HIP device: " << hipGetErrorString(err));
    }

    // Perform the copy based on direction
    hipMemcpyKind copy_kind;
    if (impl->direction_ == Direction::ToHost) {
      // GPU -> CPU
      copy_kind = hipMemcpyDeviceToHost;
      std::cerr << "MemcpyToHost: " << byte_size << " bytes" << std::endl;
    } else {
      // CPU -> GPU
      copy_kind = hipMemcpyHostToDevice;
      std::cerr << "MemcpyFromHost: " << byte_size << " bytes" << std::endl;
    }

    err = hipMemcpy(dst_data, src_data, byte_size, copy_kind);
    if (err != hipSuccess) {
      RETURN_ERROR(impl->factory_.ort_api, ORT_EP_FAIL,
                   "MemcpyKernel: hipMemcpy failed: " << hipGetErrorString(err));
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
void ORT_API_CALL MemcpyKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<MemcpyKernelImpl*>(this_ptr);
}

OrtStatus* ORT_API_CALL CreateMemcpyToHostKernel(
    void* kernel_create_func_state,
    const OrtKernelInfo* /*info*/,
    OrtKernelImpl** kernel_out) {
  auto* factory = static_cast<HipDNNEpFactory*>(kernel_create_func_state);
  *kernel_out = new MemcpyKernelImpl(*factory, MemcpyKernelImpl::Direction::ToHost, factory->GetDeviceId());
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateMemcpyFromHostKernel(
    void* kernel_create_func_state,
    const OrtKernelInfo* /*info*/,
    OrtKernelImpl** kernel_out) {
  auto* factory = static_cast<HipDNNEpFactory*>(kernel_create_func_state);
  *kernel_out = new MemcpyKernelImpl(*factory, MemcpyKernelImpl::Direction::FromHost, factory->GetDeviceId());
  return nullptr;
}

OrtStatus* RegisterMemcpyKernels(
    HipDNNEpFactory& factory,
    OrtKernelRegistry* kernel_registry,
    const char* ep_name) {
  const OrtEpApi& ep_api = factory.ep_api;

  // Get all tensor data types for type constraint
  std::vector<const OrtDataType*> all_types;
  const OrtDataType* dtype = nullptr;

  // Add common tensor types
  ONNXTensorElementDataType types_to_add[] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
  };

  for (auto elem_type : types_to_add) {
    RETURN_IF_ERROR(ep_api.GetTensorDataType(elem_type, &dtype));
    all_types.push_back(dtype);
  }

  // Register MemcpyToHost kernel (GPU -> CPU)
  {
    OrtKernelDefBuilder* builder = nullptr;
    RETURN_IF_ERROR(ep_api.CreateKernelDefBuilder(&builder));

    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetOperatorType(builder, "MemcpyToHost"));
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetDomain(builder, ""));  // ONNX domain
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetSinceVersion(builder, 1, INT_MAX));
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetExecutionProvider(builder, ep_name));

    // Input is on GPU (default), output is on CPU
    // RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetInputMemType(builder, 0, OrtMemTypeDefault));
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetOutputMemType(builder, 0, OrtMemTypeCPUOutput));

    // Add type constraint for "T"
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_AddTypeConstraint(builder, "T", all_types.data(), all_types.size()));

    OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_Build(builder, &kernel_def));
    ep_api.ReleaseKernelDefBuilder(builder);

    RETURN_IF_ERROR(ep_api.KernelRegistry_AddKernel(
        kernel_registry, kernel_def, CreateMemcpyToHostKernel, &factory));

    ep_api.ReleaseKernelDef(kernel_def);

    std::cerr << "Registered MemcpyToHost kernel for " << ep_name << std::endl;
  }

  // Register MemcpyFromHost kernel (CPU -> GPU)
  {
    OrtKernelDefBuilder* builder = nullptr;
    RETURN_IF_ERROR(ep_api.CreateKernelDefBuilder(&builder));

    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetOperatorType(builder, "MemcpyFromHost"));
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetDomain(builder, ""));  // ONNX domain
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetSinceVersion(builder, 1, INT_MAX));
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetExecutionProvider(builder, ep_name));

    // Input is on CPU, output is on GPU (default)
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetInputMemType(builder, 0, OrtMemTypeCPUInput));
    // RETURN_IF_ERROR(ep_api.KernelDefBuilder_SetOutputMemType(builder, 0, OrtMemTypeDefault));

    // Add type constraint for "T"
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_AddTypeConstraint(builder, "T", all_types.data(), all_types.size()));

    OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(ep_api.KernelDefBuilder_Build(builder, &kernel_def));
    ep_api.ReleaseKernelDefBuilder(builder);

    RETURN_IF_ERROR(ep_api.KernelRegistry_AddKernel(
        kernel_registry, kernel_def, CreateMemcpyFromHostKernel, &factory));

    ep_api.ReleaseKernelDef(kernel_def);

    std::cerr << "Registered MemcpyFromHost kernel for " << ep_name << std::endl;
  }

  return nullptr;
}

}  // namespace hipdnn_ep
