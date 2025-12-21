// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/ep_allocator.h"
#include <hip/hip_runtime.h>

namespace hipdnn_ep {

namespace {

static void StatsToKeyValuePairs(const AllocatorStats& stats, const OrtApi& api, OrtKeyValuePairs* kvps) {
  if (stats.num_allocs > 0) {
    api.AddKeyValuePair(kvps, "InUse", std::to_string(stats.bytes_in_use).c_str());
    api.AddKeyValuePair(kvps, "TotalAllocated", std::to_string(stats.total_allocated_bytes).c_str());
    api.AddKeyValuePair(kvps, "MaxInUse", std::to_string(stats.max_bytes_in_use).c_str());
    api.AddKeyValuePair(kvps, "NumAllocs", std::to_string(stats.num_allocs).c_str());
    api.AddKeyValuePair(kvps, "MaxAllocSize", std::to_string(stats.max_alloc_size).c_str());
  }
}

}  // namespace

HipDeviceAllocator::HipDeviceAllocator(const OrtMemoryInfo* mem_info, const ApiPtrs& api_ptrs, int device_id)
    : memory_info_(mem_info), api_ptrs_(api_ptrs), device_id_(device_id) {
  version = ORT_API_VERSION;
  Alloc = AllocImpl;
  Free = FreeImpl;
  Info = InfoImpl;
  Reserve = AllocImpl;  // No special reserve logic
  GetStats = GetStatsImpl;
  AllocOnStream = nullptr;  // TODO: Add stream-aware allocation
}

/*static*/
void* ORT_API_CALL HipDeviceAllocator::AllocImpl(struct OrtAllocator* this_, size_t size) {
  auto& impl = *static_cast<HipDeviceAllocator*>(this_);

  // TODO: Make allocator stream-aware. Currently using hipSetDevice which is
  // ad-hoc and affects per-thread state. Should use hipMallocAsync with a
  // device-specific stream instead.
  hipError_t err = hipSetDevice(impl.device_id_);
  if (err != hipSuccess) {
    // Can't return error from Alloc, return nullptr
    return nullptr;
  }

  void* ptr = nullptr;
  err = hipMalloc(&ptr, size);
  if (err != hipSuccess) {
    return nullptr;
  }

  // Track allocation and update stats
  {
    std::lock_guard<std::mutex> lock(impl.mutex_);
    impl.allocation_sizes_[ptr] = size;
    impl.stats_.num_allocs++;
    impl.stats_.bytes_in_use += static_cast<int64_t>(size);
    impl.stats_.total_allocated_bytes += static_cast<int64_t>(size);
    impl.stats_.max_bytes_in_use = std::max(impl.stats_.max_bytes_in_use, impl.stats_.bytes_in_use);
    impl.stats_.max_alloc_size = std::max(impl.stats_.max_alloc_size, static_cast<int64_t>(size));
  }

  return ptr;
}

/*static*/
void ORT_API_CALL HipDeviceAllocator::FreeImpl(struct OrtAllocator* this_, void* p) {
  if (p == nullptr) {
    return;
  }

  auto& impl = *static_cast<HipDeviceAllocator*>(this_);

  hipError_t err = hipSetDevice(impl.device_id_);
  if (err != hipSuccess) {
    // Can't proceed without setting the correct device
    return;
  }

  err = hipFree(p);
  if (err != hipSuccess) {
    // hipFree failed - still update our tracking to avoid memory leaks in stats
  }

  // Update tracking and stats
  {
    std::lock_guard<std::mutex> lock(impl.mutex_);
    auto it = impl.allocation_sizes_.find(p);
    if (it != impl.allocation_sizes_.end()) {
      impl.stats_.bytes_in_use -= static_cast<int64_t>(it->second);
      impl.allocation_sizes_.erase(it);
    }
  }
}

/*static*/
const struct OrtMemoryInfo* ORT_API_CALL HipDeviceAllocator::InfoImpl(const struct OrtAllocator* this_) {
  const auto& impl = *static_cast<const HipDeviceAllocator*>(this_);
  return impl.memory_info_;
}

/*static*/
OrtStatus* ORT_API_CALL HipDeviceAllocator::GetStatsImpl(const struct OrtAllocator* this_,
                                                         OrtKeyValuePairs** out) noexcept {
  const auto& impl = *static_cast<const HipDeviceAllocator*>(this_);

  OrtKeyValuePairs* kvps = nullptr;
  impl.api_ptrs_.ort_api.CreateKeyValuePairs(&kvps);

  {
    std::lock_guard<std::mutex> lock(impl.mutex_);
    StatsToKeyValuePairs(impl.stats_, impl.api_ptrs_.ort_api, kvps);
  }

  *out = kvps;
  return nullptr;
}

}  // namespace hipdnn_ep
