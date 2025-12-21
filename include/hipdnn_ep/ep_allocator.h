// Copyright (c) 2024, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_utils.h"
#include <hip/hip_runtime.h>
#include <mutex>
#include <unordered_map>

namespace hipdnn_ep {

// Allocator statistics
struct AllocatorStats {
  int64_t num_allocs{0};
  int64_t bytes_in_use{0};
  int64_t total_allocated_bytes{0};
  int64_t max_bytes_in_use{0};
  int64_t max_alloc_size{0};
};

// Base allocator with virtual destructor for proper cleanup
struct BaseAllocator : OrtAllocator {
  virtual ~BaseAllocator() = default;
};

using AllocatorUniquePtr = std::unique_ptr<BaseAllocator>;

// HIP device memory allocator
struct HipDeviceAllocator : BaseAllocator {
  HipDeviceAllocator(const OrtMemoryInfo* mem_info, const ApiPtrs& api_ptrs, int device_id);

  static void* ORT_API_CALL AllocImpl(struct OrtAllocator* this_, size_t size);
  static void ORT_API_CALL FreeImpl(struct OrtAllocator* this_, void* p);
  static const struct OrtMemoryInfo* ORT_API_CALL InfoImpl(const struct OrtAllocator* this_);
  static OrtStatus* ORT_API_CALL GetStatsImpl(const struct OrtAllocator* this_, OrtKeyValuePairs** out) noexcept;

 private:
  const OrtMemoryInfo* memory_info_;
  const ApiPtrs api_ptrs_;
  int device_id_;
  mutable std::mutex mutex_;
  std::unordered_map<void*, size_t> allocation_sizes_;
  AllocatorStats stats_;
};

}  // namespace hipdnn_ep
