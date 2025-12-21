// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/ep_factory.h"
#include "hipdnn_ep/ep.h"
#include <hip/hip_runtime.h>

namespace hipdnn_ep {

HipDNNEpFactory::HipDNNEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger)
    : OrtEpFactory{},
      ApiPtrs(apis),
      default_logger_(default_logger),
      ep_name_(ep_name),
      default_memory_info_{nullptr},
      readonly_memory_info_{nullptr} {

  ort_version_supported = ORT_API_VERSION;

  // Initialize function pointers
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVendorId = GetVendorIdImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

  // Get the first available HIP device
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err == hipSuccess && device_count > 0) {
    device_id_ = 0;  // Use first device

    hipDeviceProp_t props;
    (void)hipGetDeviceProperties(&props, device_id_);

    IGNORE_ORTSTATUS(ort_api.Logger_LogMessage(
        &default_logger_, ORT_LOGGING_LEVEL_INFO,
        ("HipDNN EP: Found GPU device: " + std::string(props.name)).c_str(),
        EP_FILE, __LINE__, __FUNCTION__));
  }

  // Setup memory info for device memory
  default_memory_info_ = Ort::MemoryInfo{
      "HipDNN_GPU",
      OrtMemoryInfoDeviceType_GPU,
      vendor_id_,
      static_cast<uint32_t>(device_id_),
      OrtDeviceMemoryType_DEFAULT,
      0,  // alignment
      OrtAllocatorType::OrtDeviceAllocator};

  readonly_memory_info_ = Ort::MemoryInfo{
      "HipDNN_GPU_readonly",
      OrtMemoryInfoDeviceType_GPU,
      vendor_id_,
      static_cast<uint32_t>(device_id_),
      OrtDeviceMemoryType_DEFAULT,
      0,
      OrtAllocatorType::OrtReadOnlyAllocator};

  // Create data transfer
  const OrtMemoryDevice* device = ep_api.MemoryInfo_GetMemoryDevice(default_memory_info_);
  data_transfer_impl_ = std::make_unique<HipDataTransfer>(apis, device, device_id_);
}

/*static*/
const char* ORT_API_CALL HipDNNEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const HipDNNEpFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

/*static*/
const char* ORT_API_CALL HipDNNEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const HipDNNEpFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

/*static*/
uint32_t ORT_API_CALL HipDNNEpFactory::GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const HipDNNEpFactory*>(this_ptr);
  return factory->vendor_id_;
}

/*static*/
const char* ORT_API_CALL HipDNNEpFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const HipDNNEpFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEpFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {

  auto* factory = static_cast<HipDNNEpFactory*>(this_ptr);
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];

    // We support GPU devices
    OrtHardwareDeviceType device_type = factory->ort_api.HardwareDevice_Type(&device);

    if (device_type == OrtHardwareDeviceType_GPU) {
      // Check if this is an AMD GPU
      // For now, we'll support any GPU since we can't easily check vendor from ORT API
      // In a real implementation, you'd check the device vendor

      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api.CreateKeyValuePairs(&ep_options);

      factory->ort_api.AddKeyValuePair(ep_metadata, "backend", "hipDNN");
      factory->ort_api.AddKeyValuePair(ep_options, "device_id", std::to_string(factory->device_id_).c_str());

      OrtEpDevice* ep_device = nullptr;
      auto* status = factory->ep_api.CreateEpDevice(factory, &device, ep_metadata, ep_options, &ep_device);

      factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      // Register allocator info
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_));
      RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->readonly_memory_info_));

      ep_devices[num_ep_devices++] = ep_device;
    }
  }

  // If no GPU was found in the device list, also check for CPU (for testing)
  // This allows the EP to work even when no explicit GPU device is provided
  if (num_ep_devices == 0) {
    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      const OrtHardwareDevice& device = *devices[i];
      OrtHardwareDeviceType device_type = factory->ort_api.HardwareDevice_Type(&device);

      if (device_type == OrtHardwareDeviceType_CPU) {
        // Accept CPU as a fallback for testing
        OrtKeyValuePairs* ep_metadata = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_metadata);
        factory->ort_api.AddKeyValuePair(ep_metadata, "backend", "hipDNN");

        OrtEpDevice* ep_device = nullptr;
        auto* status = factory->ep_api.CreateEpDevice(factory, &device, ep_metadata, nullptr, &ep_device);

        factory->ort_api.ReleaseKeyValuePairs(ep_metadata);

        if (status != nullptr) {
          return status;
        }

        RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->default_memory_info_));
        ep_devices[num_ep_devices++] = ep_device;
        break;
      }
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEpFactory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* /*devices*/,
    const OrtKeyValuePairs* const* /*ep_metadata*/,
    size_t num_devices,
    const OrtSessionOptions* session_options,
    const OrtLogger* logger,
    OrtEp** ep) noexcept {

  auto* factory = static_cast<HipDNNEpFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    return factory->ort_api.CreateStatus(
        ORT_INVALID_ARGUMENT,
        "hipDNN EP currently only supports selection for one device.");
  }

  RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(
      logger, ORT_LOGGING_LEVEL_INFO,
      "Creating hipDNN Execution Provider",
      ORT_FILE, __LINE__, __FUNCTION__));

  // Parse configuration from session options
  std::string ep_context_enable;
  RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(*session_options, "ep.context_enable", "0", ep_context_enable));

  HipDNNEp::Config config{};
  config.enable_ep_context = (ep_context_enable == "1");

  try {
    auto hipdnn_ep = std::make_unique<HipDNNEp>(*factory, config, *logger);
    *ep = hipdnn_ep.release();
  } catch (const std::exception& ex) {
    return factory->ort_api.CreateStatus(ORT_EP_FAIL, ex.what());
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL HipDNNEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  delete static_cast<HipDNNEp*>(ep);
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEpFactory::CreateAllocatorImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* /*allocator_options*/,
    OrtAllocator** allocator) noexcept {

  auto& factory = *static_cast<HipDNNEpFactory*>(this_ptr);
  std::lock_guard<std::mutex> lock(factory.mutex_);

  *allocator = nullptr;

  // Create allocator if not already created
  if (!factory.device_allocator_) {
    factory.device_allocator_ = std::make_unique<HipDeviceAllocator>(
        memory_info, factory, factory.device_id_);
  }

  *allocator = factory.device_allocator_.get();
  return nullptr;
}

/*static*/
void ORT_API_CALL HipDNNEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this_ptr*/,
                                                         OrtAllocator* /*allocator*/) noexcept {
  // Allocator is owned by factory, don't delete here
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEpFactory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr,
    OrtDataTransferImpl** data_transfer) noexcept {

  auto& factory = *static_cast<HipDNNEpFactory*>(this_ptr);
  *data_transfer = factory.data_transfer_impl_.get();
  return nullptr;
}

/*static*/
bool ORT_API_CALL HipDNNEpFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  // TODO: Implement stream support
  return false;
}

/*static*/
OrtStatus* ORT_API_CALL HipDNNEpFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* /*this_ptr*/,
    const OrtMemoryDevice* /*memory_device*/,
    const OrtKeyValuePairs* /*stream_options*/,
    OrtSyncStreamImpl** stream) noexcept {

  // TODO: Implement stream support
  *stream = nullptr;
  return nullptr;
}

}  // namespace hipdnn_ep
