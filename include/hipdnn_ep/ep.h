// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "ep_utils.h"

namespace hipdnn_ep {

class HipDNNEpFactory;
struct Kernel;

/// @brief MIOpen-based Execution Provider implementation
class HipDNNEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context{false};
    // Additional configuration options
  };

  HipDNNEp(HipDNNEpFactory& factory, const Config& config, const OrtLogger& logger);
  ~HipDNNEp();

  // Accessors
  Kernel* GetKernel(const std::string& name);
  HipDNNEpFactory& GetFactory() { return factory_; }

 private:
  // OrtEp interface implementations
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEp* this_ptr,
      const OrtMemoryInfo* memory_info,
      OrtAllocator** allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEp* this_ptr,
      const OrtMemoryDevice* memory_device,
      OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* this_ptr,
      const OrtGraph* graph,
      OrtEpGraphSupportInfo* graph_support_info) noexcept;

  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* this_ptr,
      const OrtGraph** graphs,
      const OrtNode** fused_nodes,
      size_t count,
      OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept;

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* this_ptr,
      OrtNodeComputeInfo** node_compute_infos,
      size_t num_node_compute_infos) noexcept;

  static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
      OrtEp* this_ptr,
      const OrtKernelRegistry** kernel_registry) noexcept;

  // Member data
  HipDNNEpFactory& factory_;
  Config config_;
  const OrtLogger& logger_;

  // Compiled kernels (each Kernel manages its own MIOpen handle)
  std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels_;
};

}  // namespace hipdnn_ep
