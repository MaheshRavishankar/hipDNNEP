# Plan: Add hipBLAS-LT Matrix-Multiply Support to hipDNN EP

## Overview

Add support for ONNX `MatMul` and `Gemm` operations using AMD's hipBLAS-LT library, following the existing patterns established for Conv2D with hipDNN.

## Architecture Decision

**Extend existing `Kernel` class** rather than creating a separate BlasKernel class.

Rationale:
- Follows existing code patterns
- Shared infrastructure (workspace, error handling)
- Simpler EP integration
- Future fusion potential (MatMul+Add, etc.)

Trade-off: For MatMul/Gemm, we bypass the hipDNN graph API and execute hipBLAS-LT directly since it uses an immediate execution model rather than graph-based.

## Implementation Steps

### Phase 1: Build System & Infrastructure

**1.1 CMakeLists.txt** - Add optional hipBLAS-LT detection:
```cmake
find_package(hipblaslt CONFIG)
if(hipblaslt_FOUND)
  target_link_libraries(hipdnn_ep PRIVATE hipblaslt)
  target_compile_definitions(hipdnn_ep PRIVATE HIPDNN_EP_HAS_HIPBLASLT)
endif()
```

**1.2 New file: `include/hipdnn_ep/blas_context.h`**
- `BlasContext`: RAII wrapper for `hipblasLtHandle_t`
- `MatMulPlan`: Pre-computed execution plan with matrix layouts, algorithm, workspace size

**1.3 New file: `src/blas_context.cc`**
- Handle creation/destruction
- `MatMulPlan::Initialize()`: Set up layouts based on shapes/transposes, run algorithm heuristic search

### Phase 2: Operation Support

**2.1 Modify `src/ep.cc`**
- Add `IsSupportedMatMul()`: Check 2 inputs, 2D shapes, float/float16 types
- Add `IsSupportedGemm()`: Check 2-3 inputs, handle transA/transB for dimension validation
- Update `IsSupportedOp()` dispatcher

**2.2 Modify `include/hipdnn_ep/ep.h`**
- Add `std::unique_ptr<BlasContext> blas_context_` member
- Add `GetBlasContext()` accessor

**2.3 Update EP constructor** to initialize BlasContext

### Phase 3: Kernel Implementation

**3.1 Modify `include/hipdnn_ep/kernel.h`**
- Add `BlasContext* blas_context_` member
- Add `std::unique_ptr<MatMulPlan> matmul_plan_` member
- Add `bool is_matmul_only_` flag
- Add `ExecuteMatMul()` declaration

**3.2 Modify `src/kernel.cc`**
- Add `GetFloatAttrOrDefault()` helper
- Add `AddMatMulNode()`: Create MatMulPlan for simple A @ B
- Add `AddGemmNode()`: Create MatMulPlan with transA, transB, alpha, beta
- Update `AddNode()` dispatcher
- Add `ExecuteMatMul()`: Call `hipblasLtMatmul()` with pre-computed plan
- Update `Execute()` to route MatMul/Gemm to `ExecuteMatMul()`

### Phase 4: Testing

**4.1 New file: `test/gen_matmul_model.py`**
- Generate ONNX models for MatMul and Gemm with various configs

**4.2 New file: `test/test_matmul.cc`**
- `ReferenceMatMul()`: CPU reference implementation
- Test cases: BasicMatMul, BasicGemm, GemmWithTranspose, GemmWithScaling, Float16MatMul

**4.3 Update `test/CMakeLists.txt`**
- Add test_matmul.cc to test executable
- Copy test models to build directory

## ONNX Operation Specifications

### MatMul
- Inputs: A[M,K], B[K,N]
- Output: Y[M,N] = A @ B
- No attributes

### Gemm
- Inputs: A, B, C (optional)
- Output: Y = alpha * op(A) @ op(B) + beta * C
- Attributes:
  - `transA` (int, default 0): Transpose A if 1
  - `transB` (int, default 0): Transpose B if 1
  - `alpha` (float, default 1.0): Scale for A @ B
  - `beta` (float, default 1.0): Scale for C

## Initial Limitations

- 2D matrices only (no batched matmul)
- Static shapes only
- float and float16 data types
- No broadcasting for Gemm's C input

## Files to Modify

| File | Changes |
|------|---------|
| `CMakeLists.txt` | Add hipBLAS-LT detection |
| `include/hipdnn_ep/blas_context.h` | **NEW** - BlasContext, MatMulPlan |
| `src/blas_context.cc` | **NEW** - Implementation |
| `include/hipdnn_ep/ep.h` | Add BlasContext member |
| `src/ep.cc` | Add IsSupportedMatMul/Gemm, update dispatcher |
| `include/hipdnn_ep/kernel.h` | Add MatMul members |
| `src/kernel.cc` | Add MatMul/Gemm handlers, ExecuteMatMul |
| `test/gen_matmul_model.py` | **NEW** - Test model generator |
| `test/test_matmul.cc` | **NEW** - Tests |
| `test/CMakeLists.txt` | Add new test file |

## Verification

1. **Build verification**:
   ```bash
   cmake --preset RelWithDebInfo
   cmake --build --preset RelWithDebInfo
   ```

2. **Run tests**:
   ```bash
   ctest --preset RelWithDebInfo --output-on-failure
   ```

3. **Manual verification**: Run inference session with MatMul/Gemm models and compare outputs against CPU EP with tolerance 1e-4.
