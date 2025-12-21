// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <memory>

#include "ep_utils.h"
#include "ep_allocator.h"
#include "ep_data_transfer.h"

namespace hipdnn_ep {

class HipDNNEp;  // Forward declaration

/// @brief Factory for creating hipDNN Execution Provider instances
class HipDNNEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  HipDNNEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger);
  ~HipDNNEpFactory() = default;

  // Accessors
  HipDataTransfer* GetDataTransfer() const { return data_transfer_impl_.get(); }
  int GetDeviceId() const { return device_id_; }

 private:
  // OrtEpFactory interface implementations
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      size_t num_devices,
      OrtEpDevice** ep_devices,
      size_t max_ep_devices,
      size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      const OrtKeyValuePairs* const* ep_metadata,
      size_t num_devices,
      const OrtSessionOptions* session_options,
      const OrtLogger* logger,
      OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEpFactory* this_ptr,
      const OrtMemoryInfo* memory_info,
      const OrtKeyValuePairs* allocator_options,
      OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_ptr,
      OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtMemoryDevice* memory_device,
      const OrtKeyValuePairs* stream_options,
      OrtSyncStreamImpl** stream) noexcept;

  // Member data
  const OrtLogger& default_logger_;
  const std::string ep_name_;
  const std::string vendor_{"AMD"};
  const uint32_t vendor_id_{0x1002};  // AMD PCI vendor ID
  const std::string ep_version_{"0.1.0"};

  int device_id_{0};

  // Memory info for device memory
  Ort::MemoryInfo default_memory_info_;
  Ort::MemoryInfo readonly_memory_info_;

  // Allocator management
  std::unique_ptr<HipDeviceAllocator> device_allocator_;
  std::mutex mutex_;

  // Data transfer
  std::unique_ptr<HipDataTransfer> data_transfer_impl_;
};

}  // namespace hipdnn_ep
