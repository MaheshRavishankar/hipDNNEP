// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/pointwise_graph/pointwise_graph.h"
#include "hipdnn_ep/utils/ep_utils.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>

namespace hipdnn_ep {

namespace {

/// Supported binary pointwise operation kinds.
enum class PointwiseOp {
  kMul,
  kSub,
  kAdd,
  kDiv,
};

/// Map an ONNX op type string to a PointwiseOp.
static std::optional<PointwiseOp> ToPointwiseOp(const std::string& op_type) {
  if (op_type == "Mul") return PointwiseOp::kMul;
  if (op_type == "Sub") return PointwiseOp::kSub;
  if (op_type == "Add") return PointwiseOp::kAdd;
  if (op_type == "Div") return PointwiseOp::kDiv;
  return std::nullopt;
}

// ---------- HIP kernels for binary pointwise ----------

// Block size for pointwise kernels.
constexpr unsigned kBlockSize = 256;

__global__ void MulKernel(const float* a, const float* b, float* y, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] * b[i];
  }
}

__global__ void SubKernel(const float* a, const float* b, float* y, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] - b[i];
  }
}

__global__ void AddKernel(const float* a, const float* b, float* y, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] + b[i];
  }
}

__global__ void DivKernel(const float* a, const float* b, float* y, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] / b[i];
  }
}

/// Launch the appropriate HIP kernel for the given operation.
static Status LaunchPointwiseKernel(PointwiseOp op,
                                    const float* a,
                                    const float* b,
                                    float* y,
                                    size_t n) {
  unsigned grid = static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);
  switch (op) {
    case PointwiseOp::kMul:
      MulKernel<<<grid, kBlockSize>>>(a, b, y, n);
      break;
    case PointwiseOp::kSub:
      SubKernel<<<grid, kBlockSize>>>(a, b, y, n);
      break;
    case PointwiseOp::kAdd:
      AddKernel<<<grid, kBlockSize>>>(a, b, y, n);
      break;
    case PointwiseOp::kDiv:
      DivKernel<<<grid, kBlockSize>>>(a, b, y, n);
      break;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return Status::Failure(std::string("HIP kernel launch failed: ") +
                           hipGetErrorString(err));
  }
  err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    return Status::Failure(std::string("HIP device synchronize failed: ") +
                           hipGetErrorString(err));
  }
  return Status::Success();
}

}  // namespace

//
// PointwiseGraphImpl - pimpl implementation
//

struct PointwiseGraphImpl {
  PointwiseGraphImpl() = default;

  ~PointwiseGraphImpl() = default;

  Status Build(const std::vector<Ort::ConstValueInfo>& graph_inputs,
               const std::vector<Ort::ConstValueInfo>& graph_outputs,
               const std::vector<Ort::ConstNode>& nodes);

  Status Compile();

  Status Execute(OrtKernelContext* kernel_ctx);

 private:
  PointwiseOp op_;
  std::vector<int64_t> output_shape_;
  size_t elem_count_{0};
};

Status PointwiseGraphImpl::Build(
    const std::vector<Ort::ConstValueInfo>& /*graph_inputs*/,
    const std::vector<Ort::ConstValueInfo>& graph_outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  if (nodes.size() != 1) {
    return Status::Failure("PointwiseGraph only supports single-node graphs");
  }

  std::string op_type = nodes[0].GetOperatorType();
  auto maybe_op = ToPointwiseOp(op_type);
  if (!maybe_op.has_value()) {
    return Status::Failure("Unsupported pointwise op type: " + op_type);
  }
  op_ = maybe_op.value();

  if (graph_outputs.size() != 1) {
    return Status::Failure("PointwiseGraph expects exactly 1 output");
  }

  auto shape = GetTensorShape(graph_outputs[0]);
  if (!shape.has_value()) {
    return Status::Failure("Graph output must have static shape");
  }
  output_shape_ = shape.value();
  elem_count_ = 1;
  for (auto d : output_shape_) {
    elem_count_ *= static_cast<size_t>(d);
  }

  return Status::Success();
}

Status PointwiseGraphImpl::Compile() {
  // Nothing to compile for direct HIP kernels.
  return Status::Success();
}

Status PointwiseGraphImpl::Execute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext context(kernel_ctx);

    if (context.GetInputCount() != 2) {
      return Status::Failure("Pointwise op expects 2 inputs, got " +
                             std::to_string(context.GetInputCount()));
    }

    Ort::ConstValue a_tensor = context.GetInput(0);
    Ort::ConstValue b_tensor = context.GetInput(1);

    const auto* a_ptr = a_tensor.GetTensorData<float>();
    const auto* b_ptr = b_tensor.GetTensorData<float>();

    Ort::UnownedValue output = context.GetOutput(0, output_shape_);
    auto* y_ptr = output.GetTensorMutableData<float>();

    return LaunchPointwiseKernel(op_, a_ptr, b_ptr, y_ptr, elem_count_);
  } catch (const Ort::Exception& ex) {
    return Status::Failure(std::string("ORT exception: ") + ex.what());
  }
}

//
// PointwiseGraph public interface
//

PointwiseGraph::PointwiseGraph()
    : impl_(std::make_unique<PointwiseGraphImpl>()) {}

PointwiseGraph::~PointwiseGraph() = default;

Status PointwiseGraph::Build(
    const std::vector<Ort::ConstValueInfo>& graph_inputs,
    const std::vector<Ort::ConstValueInfo>& graph_outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  return impl_->Build(graph_inputs, graph_outputs, nodes);
}

Status PointwiseGraph::Compile() {
  return impl_->Compile();
}

Status PointwiseGraph::Execute(OrtKernelContext* kernel_ctx) {
  return impl_->Execute(kernel_ctx);
}

}  // namespace hipdnn_ep
