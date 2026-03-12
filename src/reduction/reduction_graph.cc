// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/reduction/reduction_graph.h"
#include "hipdnn_ep/utils/ep_utils.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace hipdnn_ep {

namespace {

// Map ONNX op type to ReductionMode.
static std::optional<ReductionMode> GetReductionMode(const std::string& op_type) {
  if (op_type == "ReduceSum") return ReductionMode::kSum;
  if (op_type == "ReduceMax") return ReductionMode::kMax;
  if (op_type == "ReduceMin") return ReductionMode::kMin;
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// HIP reduction kernels
// ---------------------------------------------------------------------------

// Generic reduction kernel: reduces `inner_size` elements for each of
// `outer_size` independent reductions.
//
// Given an input tensor conceptually reshaped to [outer_size, inner_size],
// the kernel writes one output element per outer index by reducing along
// the inner dimension.  Thread-grid: one block per outer index, threads
// cooperate via shared memory to reduce the inner dimension.

template <typename T>
__global__ void ReduceSumKernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                int64_t outer_size,
                                int64_t inner_size) {
  extern __shared__ char shared_mem[];
  T* sdata = reinterpret_cast<T*>(shared_mem);

  int64_t outer_idx = blockIdx.x;
  if (outer_idx >= outer_size) return;

  const T* in_ptr = input + outer_idx * inner_size;

  // Each thread accumulates a partial sum over a strided range.
  T partial = static_cast<T>(0);
  for (int64_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
    partial += in_ptr[i];
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();

  // Tree reduction in shared memory.
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[outer_idx] = sdata[0];
  }
}

template <typename T>
__global__ void ReduceMaxKernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                int64_t outer_size,
                                int64_t inner_size) {
  extern __shared__ char shared_mem[];
  T* sdata = reinterpret_cast<T*>(shared_mem);

  int64_t outer_idx = blockIdx.x;
  if (outer_idx >= outer_size) return;

  const T* in_ptr = input + outer_idx * inner_size;

  T partial = in_ptr[0];
  for (int64_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
    T val = in_ptr[i];
    partial = (val > partial) ? val : partial;
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();

  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      T other = sdata[threadIdx.x + s];
      if (other > sdata[threadIdx.x]) {
        sdata[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[outer_idx] = sdata[0];
  }
}

template <typename T>
__global__ void ReduceMinKernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                int64_t outer_size,
                                int64_t inner_size) {
  extern __shared__ char shared_mem[];
  T* sdata = reinterpret_cast<T*>(shared_mem);

  int64_t outer_idx = blockIdx.x;
  if (outer_idx >= outer_size) return;

  const T* in_ptr = input + outer_idx * inner_size;

  T partial = in_ptr[0];
  for (int64_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
    T val = in_ptr[i];
    partial = (val < partial) ? val : partial;
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();

  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      T other = sdata[threadIdx.x + s];
      if (other < sdata[threadIdx.x]) {
        sdata[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    output[outer_idx] = sdata[0];
  }
}

// Transpose (permute) kernel: reorder elements according to a permutation
// so that the reduction axes become the innermost (rightmost) dimensions.
// This kernel handles arbitrary rank tensors.
//
// Each thread computes one output element.  The flat output index is
// decomposed into multi-dimensional coordinates of the *output* tensor,
// which are then mapped to the *input* via the inverse permutation to
// obtain the source flat index.
template <typename T>
__global__ void PermuteKernel(const T* __restrict__ input,
                              T* __restrict__ output,
                              const int64_t* __restrict__ out_shape,
                              const int64_t* __restrict__ out_strides,
                              const int64_t* __restrict__ in_strides_permuted,
                              int rank,
                              int64_t total_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  // Decompose flat output index into multi-dim coords, then compute
  // the corresponding flat input index.
  int64_t in_idx = 0;
  int64_t remaining = idx;
  for (int d = 0; d < rank; ++d) {
    int64_t coord = remaining / out_strides[d];
    remaining -= coord * out_strides[d];
    in_idx += coord * in_strides_permuted[d];
  }
  output[idx] = input[in_idx];
}

// Launch a transpose kernel so that the reduction axes become contiguous
// at the end of the shape.  Returns the pointer to the transposed data
// (which lives in a newly allocated device buffer owned by the caller).
template <typename T>
static Status LaunchPermute(const T* input,
                            T** output,
                            const std::vector<int64_t>& in_shape,
                            const std::vector<int64_t>& perm) {
  int rank = static_cast<int>(in_shape.size());
  int64_t total = 1;
  for (int64_t d : in_shape) total *= d;

  // Compute input strides (row-major).
  std::vector<int64_t> in_strides(rank);
  {
    int64_t s = 1;
    for (int i = rank - 1; i >= 0; --i) {
      in_strides[i] = s;
      s *= in_shape[i];
    }
  }

  // Output shape after permutation.
  std::vector<int64_t> out_shape(rank);
  for (int i = 0; i < rank; ++i) out_shape[i] = in_shape[perm[i]];

  // Output strides (row-major).
  std::vector<int64_t> out_strides(rank);
  {
    int64_t s = 1;
    for (int i = rank - 1; i >= 0; --i) {
      out_strides[i] = s;
      s *= out_shape[i];
    }
  }

  // in_strides_permuted[d] = in_strides[perm[d]].
  // This lets us go from an output coordinate to the corresponding input
  // flat offset.
  std::vector<int64_t> in_strides_permuted(rank);
  for (int i = 0; i < rank; ++i) {
    in_strides_permuted[i] = in_strides[perm[i]];
  }

  // RAII wrapper for device memory to avoid leak-prone manual cleanup.
  struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() { if (ptr) (void)hipFree(ptr); }
  };

  size_t meta_bytes = rank * sizeof(int64_t);

  DeviceBuffer d_out_shape_buf, d_out_strides_buf, d_in_strides_permuted_buf;
  hipError_t err;
  err = hipMalloc(&d_out_shape_buf.ptr, meta_bytes);
  if (err != hipSuccess) return Status::Failure("hipMalloc failed");
  err = hipMalloc(&d_out_strides_buf.ptr, meta_bytes);
  if (err != hipSuccess) return Status::Failure("hipMalloc failed");
  err = hipMalloc(&d_in_strides_permuted_buf.ptr, meta_bytes);
  if (err != hipSuccess) return Status::Failure("hipMalloc failed");

  auto* d_out_shape = static_cast<int64_t*>(d_out_shape_buf.ptr);
  auto* d_out_strides = static_cast<int64_t*>(d_out_strides_buf.ptr);
  auto* d_in_strides_permuted = static_cast<int64_t*>(d_in_strides_permuted_buf.ptr);

  err = hipMemcpy(d_out_shape, out_shape.data(), meta_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) return Status::Failure("hipMemcpy failed");
  err = hipMemcpy(d_out_strides, out_strides.data(), meta_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) return Status::Failure("hipMemcpy failed");
  err = hipMemcpy(d_in_strides_permuted, in_strides_permuted.data(), meta_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) return Status::Failure("hipMemcpy failed");

  // Allocate output buffer (caller takes ownership via *output).
  T* d_output = nullptr;
  err = hipMalloc(&d_output, total * sizeof(T));
  if (err != hipSuccess) return Status::Failure("hipMalloc failed for permute output");

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  PermuteKernel<T><<<blocks, kThreads>>>(
      input, d_output, d_out_shape, d_out_strides,
      d_in_strides_permuted, rank, total);

  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    (void)hipFree(d_output);
    return Status::Failure("hipDeviceSynchronize failed after permute");
  }

  *output = d_output;
  return Status::Success();
}

// Launch the appropriate reduction kernel.
template <typename T>
static Status LaunchReduction(ReductionMode mode,
                              const T* input,
                              T* output,
                              int64_t outer_size,
                              int64_t inner_size) {
  constexpr int kMaxThreads = 256;
  int threads = std::min(static_cast<int>(inner_size),
                         kMaxThreads);
  // Round up to next power of two for the tree reduction.
  int po2 = 1;
  while (po2 < threads) po2 <<= 1;
  threads = po2;

  size_t shared_bytes = threads * sizeof(T);
  int blocks = static_cast<int>(outer_size);

  switch (mode) {
    case ReductionMode::kSum:
      ReduceSumKernel<T><<<blocks, threads, shared_bytes>>>(
          input, output, outer_size, inner_size);
      break;
    case ReductionMode::kMax:
      ReduceMaxKernel<T><<<blocks, threads, shared_bytes>>>(
          input, output, outer_size, inner_size);
      break;
    case ReductionMode::kMin:
      ReduceMinKernel<T><<<blocks, threads, shared_bytes>>>(
          input, output, outer_size, inner_size);
      break;
  }

  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    return Status::Failure(
        std::string("HIP reduction kernel failed: ") + hipGetErrorString(err));
  }
  return Status::Success();
}

}  // namespace

// ---------------------------------------------------------------------------
// ReductionGraphImpl
// ---------------------------------------------------------------------------

struct ReductionGraphImpl {
  Status Build(const std::vector<Ort::ConstValueInfo>& graph_inputs,
               const std::vector<Ort::ConstValueInfo>& graph_outputs,
               const std::vector<Ort::ConstNode>& nodes);

  Status Compile() { return Status::Success(); }

  Status Execute(OrtKernelContext* kernel_ctx);

 private:
  ReductionMode mode_;

  // Input/output metadata.
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  ONNXTensorElementDataType dtype_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};

  // Normalized reduction axes (non-negative, sorted).
  std::vector<int64_t> axes_;

  // Whether the reduction axes are already contiguous at the tail end.
  // If true, we skip the permutation and reduce directly.
  bool axes_are_contiguous_{false};

  // Permutation that moves reduction axes to the rightmost positions.
  // Only used when axes_are_contiguous_ == false.
  std::vector<int64_t> perm_;

  // Factored sizes: outer_size_ * inner_size_ == total elements.
  // inner_size_ is the product of the reduction axis extents.
  int64_t outer_size_{1};
  int64_t inner_size_{1};
};

Status ReductionGraphImpl::Build(
    const std::vector<Ort::ConstValueInfo>& graph_inputs,
    const std::vector<Ort::ConstValueInfo>& graph_outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  if (nodes.size() != 1) {
    return Status::Failure(
        "ReductionGraph expects exactly one node, got " +
        std::to_string(nodes.size()));
  }

  const auto& node = nodes[0];
  std::string op_type = node.GetOperatorType();

  auto opt_mode = GetReductionMode(op_type);
  if (!opt_mode.has_value()) {
    return Status::Failure("Unsupported reduction op: " + op_type);
  }
  mode_ = opt_mode.value();

  // Input X is always the first input.
  auto inputs = node.GetInputs();
  auto outputs = node.GetOutputs();
  if (inputs.empty() || outputs.empty()) {
    return Status::Failure("Reduction node has no inputs/outputs");
  }

  dtype_ = GetTensorElementType(inputs[0]);
  auto opt_shape = GetTensorShape(inputs[0]);
  if (!opt_shape.has_value()) {
    return Status::Failure("Input must have static shape");
  }
  input_shape_ = opt_shape.value();

  auto opt_out_shape = GetTensorShape(outputs[0]);
  if (!opt_out_shape.has_value()) {
    return Status::Failure("Output must have static shape");
  }
  output_shape_ = opt_out_shape.value();

  int64_t rank = static_cast<int64_t>(input_shape_.size());

  // Parse axes.
  // ONNX ReduceSum opset >= 13 passes axes as a second input tensor;
  // opset < 13 uses an attribute.  The EP validator already checked that
  // axes are available (either as a constant initializer or attribute).
  axes_.clear();

  // Try the "axes" attribute (opset < 13).
  axes_ = GetIntsAttrOrDefault(node, "axes", {});

  // If axes are empty, check for a second input (opset >= 13).
  if (axes_.empty() && inputs.size() >= 2) {
    // The axes tensor should be a constant initializer.
    Ort::ConstValue axes_tensor{nullptr};
    Ort::Status status = inputs[1].GetInitializer(axes_tensor);
    if (status.IsOK() && axes_tensor != nullptr) {
      auto axes_shape = GetTensorShape(inputs[1]);
      if (axes_shape.has_value()) {
        size_t n = 1;
        for (int64_t d : axes_shape.value()) n *= d;
        const int64_t* axes_data = axes_tensor.GetTensorData<int64_t>();
        if (axes_data != nullptr) {
          axes_.assign(axes_data, axes_data + n);
        }
      }
    }
  }

  // noop_with_empty_axes: when true, empty axes means identity (no reduction).
  int64_t noop = GetIntAttrOrDefault(node, "noop_with_empty_axes", 0);
  if (axes_.empty()) {
    if (noop != 0) {
      // Identity — output == input.  We'll just copy in Execute().
      // Set outer_size = total, inner_size = 1 which makes the kernel a copy.
      outer_size_ = 1;
      for (int64_t d : input_shape_) outer_size_ *= d;
      inner_size_ = 1;
      axes_are_contiguous_ = true;
      return Status::Success();
    }
    // Default: reduce over all axes.
    for (int64_t i = 0; i < rank; ++i) axes_.push_back(i);
  }

  // Normalize negative axes.
  for (auto& a : axes_) {
    if (a < 0) a += rank;
  }
  std::sort(axes_.begin(), axes_.end());

  // Check if axes are contiguous at the end of the shape.
  axes_are_contiguous_ = true;
  int64_t expected = rank - static_cast<int64_t>(axes_.size());
  for (size_t i = 0; i < axes_.size(); ++i) {
    if (axes_[i] != expected + static_cast<int64_t>(i)) {
      axes_are_contiguous_ = false;
      break;
    }
  }

  // Build the permutation that moves non-reduction axes first, then
  // reduction axes.
  if (!axes_are_contiguous_) {
    std::vector<bool> is_reduced(rank, false);
    for (int64_t a : axes_) is_reduced[a] = true;
    perm_.clear();
    for (int64_t i = 0; i < rank; ++i) {
      if (!is_reduced[i]) perm_.push_back(i);
    }
    for (int64_t a : axes_) perm_.push_back(a);
  }

  // Compute outer_size (non-reduced dims) and inner_size (reduced dims).
  outer_size_ = 1;
  inner_size_ = 1;
  std::vector<bool> is_reduced(rank, false);
  for (int64_t a : axes_) is_reduced[a] = true;
  for (int64_t i = 0; i < rank; ++i) {
    if (is_reduced[i]) {
      inner_size_ *= input_shape_[i];
    } else {
      outer_size_ *= input_shape_[i];
    }
  }

  return Status::Success();
}

Status ReductionGraphImpl::Execute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext context(kernel_ctx);
    Ort::ConstValue input = context.GetInput(0);
    const void* input_data = input.GetTensorRawData();

    Ort::UnownedValue output = context.GetOutput(0, output_shape_);
    void* output_data = output.GetTensorMutableRawData();

    if (dtype_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const float* in = static_cast<const float*>(input_data);
      float* out = static_cast<float*>(output_data);

      if (!axes_are_contiguous_) {
        // Permute so reduction axes are contiguous at the end.
        float* permuted = nullptr;
        auto status = LaunchPermute<float>(in, &permuted, input_shape_, perm_);
        if (status.failed()) return status;
        status = LaunchReduction<float>(mode_, permuted, out,
                                        outer_size_, inner_size_);
        (void)hipFree(permuted);
        return status;
      }
      return LaunchReduction<float>(mode_, in, out,
                                    outer_size_, inner_size_);
    }

    return Status::Failure("Unsupported data type for reduction");

  } catch (const Ort::Exception& ex) {
    return Status::Failure(std::string("ORT exception: ") + ex.what());
  }
}

// ---------------------------------------------------------------------------
// ReductionGraph public interface
// ---------------------------------------------------------------------------

ReductionGraph::ReductionGraph()
    : impl_(std::make_unique<ReductionGraphImpl>()) {}

ReductionGraph::~ReductionGraph() = default;

Status ReductionGraph::Build(
    const std::vector<Ort::ConstValueInfo>& graph_inputs,
    const std::vector<Ort::ConstValueInfo>& graph_outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  return impl_->Build(graph_inputs, graph_outputs, nodes);
}

Status ReductionGraph::Compile() {
  return impl_->Compile();
}

Status ReductionGraph::Execute(OrtKernelContext* kernel_ctx) {
  return impl_->Execute(kernel_ctx);
}

}  // namespace hipdnn_ep
