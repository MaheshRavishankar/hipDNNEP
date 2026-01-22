// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/blas_graph/blas_graph.h"

#ifdef HIPDNN_EP_HAS_HIPBLASLT

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

namespace hipdnn_ep {

namespace {

// Maximum workspace size to request for algorithm search (256 MB)
constexpr size_t kMaxWorkspaceSize = 256 * 1024 * 1024;

// Helper macro for hipBLAS-LT error checking
#define HIPBLASLT_CHECK(ort_api, expr)                                                       \
  do {                                                                                       \
    hipblasStatus_t _status = (expr);                                                        \
    if (_status != HIPBLAS_STATUS_SUCCESS) {                                                 \
      RETURN_ERROR(ort_api, ORT_EP_FAIL, "hipBLAS-LT error: " << static_cast<int>(_status)); \
    }                                                                                        \
  } while (0)

// Check if ONNX tensor element data type is float16
bool IsFloat16(ONNXTensorElementDataType onnx_dtype) {
  return onnx_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

// Check if data type is supported
bool IsSupportedDataType(ONNXTensorElementDataType onnx_dtype) {
  return onnx_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         onnx_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

}  // namespace

//
// BlasGraphImpl - contains all hipBLAS-LT specific types
//

struct BlasGraphImpl {
  BlasGraphImpl(const OrtApi& ort_api, hipblasLtHandle_t handle)
      : ort_api_(ort_api), handle_(handle) {}

  ~BlasGraphImpl() {
    if (preference_) {
      hipblasLtMatmulPreferenceDestroy(preference_);
    }
    if (layout_d_) {
      hipblasLtMatrixLayoutDestroy(layout_d_);
    }
    if (layout_c_) {
      hipblasLtMatrixLayoutDestroy(layout_c_);
    }
    if (layout_b_) {
      hipblasLtMatrixLayoutDestroy(layout_b_);
    }
    if (layout_a_) {
      hipblasLtMatrixLayoutDestroy(layout_a_);
    }
    if (matmul_desc_) {
      hipblasLtMatmulDescDestroy(matmul_desc_);
    }
  }

  OrtStatus* Initialize(int64_t m, int64_t n, int64_t k,
                        bool is_float16, bool trans_a, bool trans_b,
                        float alpha, float beta, bool has_bias) {
    alpha_ = alpha;
    beta_ = has_bias ? beta : 0.0f;
    has_bias_ = has_bias;

    // Determine data type
    hipDataType dtype = is_float16 ? HIP_R_16F : HIP_R_32F;

    // Compute type is always float32 for float/half inputs
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
    hipDataType scale_type = HIP_R_32F;

    // Create matmul descriptor
    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatmulDescCreate(&matmul_desc_, compute_type, scale_type));

    // Set transpose operations
    hipblasOperation_t op_a = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t op_b = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatmulDescSetAttribute(matmul_desc_,
                                                    HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &op_a, sizeof(op_a)));
    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatmulDescSetAttribute(matmul_desc_,
                                                    HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &op_b, sizeof(op_b)));

    // Matrix dimensions for layout creation
    // After transpose: A is (m, k), B is (k, n), C and D are (m, n)
    // Before transpose (storage layout):
    //   A: trans_a ? (k, m) : (m, k)
    //   B: trans_b ? (n, k) : (k, n)
    int64_t a_rows = trans_a ? k : m;
    int64_t a_cols = trans_a ? m : k;
    int64_t b_rows = trans_b ? n : k;
    int64_t b_cols = trans_b ? k : n;

    // Leading dimensions (column-major storage)
    int64_t lda = a_rows;
    int64_t ldb = b_rows;
    int64_t ldc = m;
    int64_t ldd = m;

    // Create matrix layouts
    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatrixLayoutCreate(&layout_a_, dtype, a_rows, a_cols, lda));
    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatrixLayoutCreate(&layout_b_, dtype, b_rows, b_cols, ldb));
    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatrixLayoutCreate(&layout_c_, dtype, m, n, ldc));
    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatrixLayoutCreate(&layout_d_, dtype, m, n, ldd));

    // Create preference and set workspace size limit
    HIPBLASLT_CHECK(ort_api_, hipblasLtMatmulPreferenceCreate(&preference_));
    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatmulPreferenceSetAttribute(preference_,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &kMaxWorkspaceSize,
                                                          sizeof(kMaxWorkspaceSize)));

    // Get algorithm heuristics
    int returned_results = 0;
    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatmulAlgoGetHeuristic(handle_,
                                                    matmul_desc_,
                                                    layout_a_,
                                                    layout_b_,
                                                    layout_c_,
                                                    layout_d_,
                                                    preference_,
                                                    1,  // Request 1 algorithm
                                                    &heuristic_result_,
                                                    &returned_results));

    if (returned_results == 0) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "hipBLAS-LT: No algorithm found for MatMul");
    }

    // Allocate workspace if needed
    if (heuristic_result_.workspaceSize > 0) {
      workspace_.resize(heuristic_result_.workspaceSize);
    }

    initialized_ = true;
    return nullptr;
  }

  OrtStatus* Execute(const void* a, const void* b, const void* c, void* d) {
    if (!initialized_) {
      RETURN_ERROR(ort_api_, ORT_EP_FAIL, "BlasGraph not initialized");
    }

    // If no bias, c points to d (in-place, but beta=0 so it doesn't matter)
    const void* c_ptr = (c != nullptr) ? c : d;
    void* workspace_ptr = workspace_.empty() ? nullptr : workspace_.data();

    HIPBLASLT_CHECK(ort_api_,
                    hipblasLtMatmul(handle_,
                                    matmul_desc_,
                                    &alpha_,
                                    a,
                                    layout_a_,
                                    b,
                                    layout_b_,
                                    &beta_,
                                    c_ptr,
                                    layout_c_,
                                    d,
                                    layout_d_,
                                    &heuristic_result_.algo,
                                    workspace_ptr,
                                    heuristic_result_.workspaceSize,
                                    nullptr));  // default stream

    return nullptr;
  }

  // References
  const OrtApi& ort_api_;
  hipblasLtHandle_t handle_;

  // hipBLAS-LT descriptors
  hipblasLtMatmulDesc_t matmul_desc_{nullptr};
  hipblasLtMatrixLayout_t layout_a_{nullptr};
  hipblasLtMatrixLayout_t layout_b_{nullptr};
  hipblasLtMatrixLayout_t layout_c_{nullptr};
  hipblasLtMatrixLayout_t layout_d_{nullptr};
  hipblasLtMatmulPreference_t preference_{nullptr};
  hipblasLtMatmulHeuristicResult_t heuristic_result_{};

  // Execution state
  float alpha_{1.0f};
  float beta_{0.0f};
  std::vector<char> workspace_;
  std::vector<int64_t> output_shape_;
  bool has_bias_{false};
  bool initialized_{false};
};

//
// BlasGraph implementation
//

BlasGraph::BlasGraph(const OrtApi& ort_api, hipblaslt_handle_t handle)
    : impl_(std::make_unique<BlasGraphImpl>(ort_api, static_cast<hipblasLtHandle_t>(handle))) {
}

BlasGraph::~BlasGraph() = default;

OrtStatus* BlasGraph::Build(
    [[maybe_unused]] const std::vector<Ort::ConstValueInfo>& graph_inputs,
    const std::vector<Ort::ConstValueInfo>& graph_outputs,
    const std::vector<Ort::ConstNode>& nodes) {
  // Only handle single MatMul/Gemm node graphs
  if (nodes.size() != 1) {
    RETURN_ERROR(impl_->ort_api_, ORT_EP_FAIL, "BlasGraph only supports single-node graphs");
  }

  Ort::ConstNode node = nodes[0];
  std::string op_type = node.GetOperatorType();
  if (op_type != "MatMul" && op_type != "Gemm") {
    RETURN_ERROR(impl_->ort_api_, ORT_EP_FAIL, "BlasGraph only supports MatMul and Gemm ops");
  }

  // Extract output shape
  auto output_shape = GetTensorShape(graph_outputs[0]);
  if (!output_shape.has_value()) {
    RETURN_ERROR(impl_->ort_api_, ORT_EP_FAIL, "Graph output must have static shape");
  }
  impl_->output_shape_ = output_shape.value();

  std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();

  // Get data type
  ONNXTensorElementDataType dtype = GetTensorElementType(inputs[0]);
  if (!IsSupportedDataType(dtype)) {
    RETURN_ERROR(impl_->ort_api_, ORT_EP_FAIL, "Unsupported data type for MatMul/Gemm");
  }
  bool is_float16 = IsFloat16(dtype);

  // Get shapes
  auto a_shape = GetTensorShape(inputs[0]);
  auto b_shape = GetTensorShape(inputs[1]);
  if (!a_shape.has_value() || !b_shape.has_value()) {
    RETURN_ERROR(impl_->ort_api_, ORT_EP_FAIL, "MatMul/Gemm requires static shapes");
  }

  // Determine M, N, K and transpose flags
  int64_t m, n, k;
  bool trans_a = false;
  bool trans_b = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool has_bias = false;

  if (op_type == "MatMul") {
    // MatMul: A[M,K] @ B[K,N] = Y[M,N] (row-major)
    // For row-major to column-major: compute B^T @ A^T = (A @ B)^T
    // Swap m and n for the BLAS call (we pass B, A instead of A, B)
    m = (*a_shape)[0];
    k = (*a_shape)[1];
    n = (*b_shape)[1];
  } else {
    // Gemm: Y = alpha * op(A) @ op(B) + beta * C
    // ONNX transpose flags apply to the logical operation, but we also
    // need to account for row-major to column-major conversion
    bool onnx_trans_a = GetIntAttrOrDefault(node, "transA", 0) != 0;
    bool onnx_trans_b = GetIntAttrOrDefault(node, "transB", 0) != 0;
    alpha = GetFloatAttrOrDefault(node, "alpha", 1.0f);
    beta = GetFloatAttrOrDefault(node, "beta", 1.0f);

    // After ONNX transpose: A is (M, K), B is (K, N)
    m = onnx_trans_a ? (*a_shape)[1] : (*a_shape)[0];
    k = onnx_trans_a ? (*a_shape)[0] : (*a_shape)[1];
    n = onnx_trans_b ? (*b_shape)[0] : (*b_shape)[1];

    // For row-major BLAS with swapped operands (B @ A instead of A @ B):
    // - If ONNX wants transA, that becomes transB for the swapped operation
    // - If ONNX wants transB, that becomes transA for the swapped operation
    trans_a = onnx_trans_b;
    trans_b = onnx_trans_a;

    has_bias = (inputs.size() == 3);
  }

  // Swap m and n for BLAS (we compute B @ A with dimensions [N,K] @ [K,M] = [N,M])
  return impl_->Initialize(n, m, k, is_float16, trans_a, trans_b, alpha, beta, has_bias);
}

OrtStatus* BlasGraph::Execute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext context(kernel_ctx);

    // Get input tensors
    size_t input_count = context.GetInputCount();
    if (input_count < 2 || input_count > 3) {
      RETURN_ERROR(impl_->ort_api_, ORT_EP_FAIL,
                   "MatMul/Gemm expects 2-3 inputs, got " << input_count);
    }

    Ort::ConstValue a_tensor = context.GetInput(0);
    Ort::ConstValue b_tensor = context.GetInput(1);
    const void* a_ptr = a_tensor.GetTensorRawData();
    const void* b_ptr = b_tensor.GetTensorRawData();

    // Get optional bias (C) for Gemm
    const void* c_ptr = nullptr;
    if (impl_->has_bias_ && input_count == 3) {
      Ort::ConstValue c_tensor = context.GetInput(2);
      c_ptr = c_tensor.GetTensorRawData();
    }

    // Allocate output
    Ort::UnownedValue output = context.GetOutput(0, impl_->output_shape_);
    void* d_ptr = output.GetTensorMutableRawData();

    // Swap A and B for row-major to column-major conversion
    // We compute B @ A instead of A @ B to get correct row-major result
    return impl_->Execute(b_ptr, a_ptr, c_ptr, d_ptr);

  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  }
}

//
// Handle creation/destruction helpers
//

hipblaslt_handle_t CreateHipBlasLtHandle() {
  hipblasLtHandle_t handle = nullptr;
  hipblasStatus_t status = hipblasLtCreate(&handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  return static_cast<hipblaslt_handle_t>(handle);
}

void DestroyHipBlasLtHandle(hipblaslt_handle_t handle) {
  if (handle != nullptr) {
    hipblasLtDestroy(static_cast<hipblasLtHandle_t>(handle));
  }
}

}  // namespace hipdnn_ep

#else  // !HIPDNN_EP_HAS_HIPBLASLT

namespace hipdnn_ep {

hipblaslt_handle_t CreateHipBlasLtHandle() {
  return nullptr;
}

void DestroyHipBlasLtHandle(hipblaslt_handle_t /*handle*/) {}

}  // namespace hipdnn_ep

#endif  // HIPDNN_EP_HAS_HIPBLASLT
