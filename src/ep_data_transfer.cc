// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/ep_data_transfer.h"
#include <hip/hip_runtime.h>
#include <cstring>

namespace hipdnn_ep {

HipDataTransfer::HipDataTransfer(ApiPtrs api_ptrs, const OrtMemoryDevice* device_mem_info, int device_id)
    : ApiPtrs(api_ptrs), device_mem_info_(device_mem_info), device_id_(device_id) {
  CanCopy = CanCopyImpl;
  CopyTensors = CopyTensorsImpl;
  Release = ReleaseImpl;
}

/*static*/
bool ORT_API_CALL HipDataTransfer::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                               const OrtMemoryDevice* src_memory_device,
                                               const OrtMemoryDevice* dst_memory_device) noexcept {
  const auto& impl = *static_cast<const HipDataTransfer*>(this_ptr);

  // Get memory types
  OrtDeviceMemoryType src_type = impl.ep_api.MemoryDevice_GetMemoryType(src_memory_device);
  OrtDeviceMemoryType dst_type = impl.ep_api.MemoryDevice_GetMemoryType(dst_memory_device);

  // We support:
  // - CPU to GPU (DEFAULT)
  // - GPU (DEFAULT) to CPU
  // - GPU to GPU on same device

  OrtMemoryInfoDeviceType src_device_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
  OrtMemoryInfoDeviceType dst_device_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);

  bool src_is_cpu = (src_device_type == OrtMemoryInfoDeviceType_CPU);
  bool dst_is_cpu = (dst_device_type == OrtMemoryInfoDeviceType_CPU);

  // CPU to GPU
  if (src_is_cpu && !dst_is_cpu && dst_type == OrtDeviceMemoryType_DEFAULT) {
    return true;
  }

  // GPU to CPU
  if (!src_is_cpu && src_type == OrtDeviceMemoryType_DEFAULT && dst_is_cpu) {
    return true;
  }

  // GPU to GPU (same device)
  if (!src_is_cpu && !dst_is_cpu &&
      src_type == OrtDeviceMemoryType_DEFAULT &&
      dst_type == OrtDeviceMemoryType_DEFAULT) {
    return true;
  }

  return false;
}

/*static*/
OrtStatus* ORT_API_CALL HipDataTransfer::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                         const OrtValue** src_tensors_ptr,
                                                         OrtValue** dst_tensors_ptr,
                                                         OrtSyncStream** /*streams_ptr*/,
                                                         size_t num_tensors) noexcept {
  auto& impl = *static_cast<HipDataTransfer*>(this_ptr);

  hipError_t err = hipSetDevice(impl.device_id_);
  if (err != hipSuccess) {
    RETURN_ERROR(impl.ort_api, ORT_EP_FAIL, "Failed to set HIP device: " << hipGetErrorString(err));
  }

  for (size_t i = 0; i < num_tensors; ++i) {
    try {
      Ort::ConstValue src{src_tensors_ptr[i]};
      Ort::UnownedValue dst{dst_tensors_ptr[i]};

      auto src_type_shape = src.GetTensorTypeAndShapeInfo();
      auto dst_type_shape = dst.GetTensorTypeAndShapeInfo();

      size_t src_size = src_type_shape.GetElementCount();
      size_t dst_size = dst_type_shape.GetElementCount();

      if (src_size != dst_size) {
        RETURN_ERROR(impl.ort_api, ORT_EP_FAIL, "Source and destination tensor sizes don't match");
      }

      // Get element size based on data type
      ONNXTensorElementDataType elem_type = src_type_shape.GetElementType();
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
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
          elem_size = sizeof(int32_t);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
          elem_size = sizeof(int64_t);
          break;
        default:
          RETURN_ERROR(impl.ort_api, ORT_EP_FAIL, "Unsupported tensor element type");
      }

      size_t byte_size = src_size * elem_size;

      // Get memory info for source and destination
      Ort::ConstMemoryInfo src_mem_info = src.GetTensorMemoryInfo();
      Ort::ConstMemoryInfo dst_mem_info = dst.GetTensorMemoryInfo();

      OrtMemoryInfoDeviceType src_device_type = src_mem_info.GetDeviceType();
      OrtMemoryInfoDeviceType dst_device_type = dst_mem_info.GetDeviceType();

      const void* src_data = src.GetTensorRawData();
      void* dst_data = dst.GetTensorMutableRawData();

      hipMemcpyKind kind;
      if (src_device_type == OrtMemoryInfoDeviceType_CPU &&
          dst_device_type == OrtMemoryInfoDeviceType_GPU) {
        kind = hipMemcpyHostToDevice;
      } else if (src_device_type == OrtMemoryInfoDeviceType_GPU &&
                 dst_device_type == OrtMemoryInfoDeviceType_CPU) {
        kind = hipMemcpyDeviceToHost;
      } else if (src_device_type == OrtMemoryInfoDeviceType_GPU &&
                 dst_device_type == OrtMemoryInfoDeviceType_GPU) {
        kind = hipMemcpyDeviceToDevice;
      } else {
        // CPU to CPU - use memcpy
        std::memcpy(dst_data, src_data, byte_size);
        continue;
      }

      // TODO: Use async copy with streams when available
      err = hipMemcpy(dst_data, src_data, byte_size, kind);
      if (err != hipSuccess) {
        RETURN_ERROR(impl.ort_api, ORT_EP_FAIL, "hipMemcpy failed: " << hipGetErrorString(err));
      }

    } catch (const Ort::Exception& ex) {
      Ort::Status status(ex);
      return status.release();
    } catch (const std::exception& ex) {
      Ort::Status status(ex.what(), ORT_EP_FAIL);
      return status.release();
    }
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL HipDataTransfer::ReleaseImpl(OrtDataTransferImpl* /*this_ptr*/) noexcept {
  // Data transfer is owned by factory, don't delete here
}

}  // namespace hipdnn_ep
