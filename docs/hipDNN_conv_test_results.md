# hipDNN Convolution Test Results

**Test Date:** January 15, 2026  
**Test Executable:** `hipdnn_conv_tests.exe`  
**Platform:** Windows 10, AMD GPU (gfx1151)

## Summary

| Test Name | Status | Duration | Notes |
|-----------|--------|----------|-------|
| ConvolutionWithoutBias | ✅ PASSED | 657 ms | Max diff: 1.19e-06 |
| ConvolutionWithBias | ❌ FAILED | 14 ms | No engine configurations |

**Overall:** 1 passed, 1 failed

---

## Test Details

### Test 1: ConvolutionWithoutBias ✅

**Status:** PASSED

**Configuration:**
- Input: `[1, 3, 8, 8]` (N, C_in, H, W)
- Weights: `[2, 3, 3, 3]` (C_out, C_in, K_h, K_w)
- Output: `[1, 2, 8, 8]`
- Padding: `[1, 1]`
- Stride: `[1, 1]`
- Dilation: `[1, 1]`
- Data Type: FLOAT

**Execution Flow:**
1. ✅ Graph validation passed
2. ✅ Operation graph built
3. ✅ Execution plans created
4. ✅ Plans built
5. ✅ Execution completed

**Result:**
- Max numerical difference vs CPU reference: **1.19209e-06**
- Within tolerance (1e-4)

---

### Test 2: ConvolutionWithBias ❌

**Status:** FAILED

**Configuration:**
- Input: `[1, 3, 8, 8]`
- Weights: `[2, 3, 3, 3]`
- Bias: `[1, 2, 1, 1]` (broadcast shape)
- Output: `[1, 2, 8, 8]`
- Uses virtual intermediate tensor for Conv → Pointwise ADD fusion

**Execution Flow:**
1. ✅ Graph validation passed
2. ✅ Operation graph built
3. ❌ **Execution plans creation FAILED**

**Error:**
```
hipDNN create_execution_plans failed: No engine configurations available for the graph.
```

---

## Analysis

### Why ConvolutionWithBias Failed

The hipDNN frontend's graph API attempts to fuse the convolution and pointwise ADD (bias) operations into a single optimized kernel. However, **no engine (backend implementation) is available** for this specific fused pattern.

**Implications:**
1. hipDNN's MIOpen backend does not support `Conv + PointwiseADD` fusion
2. Standalone convolution works correctly
3. For Conv+Bias, alternative approaches are needed:
   - Use MIOpen directly with separate `miopenConvolutionForward` + `miopenOpTensor` calls
   - Execute two separate hipDNN graphs (Conv graph, then Bias graph)

### Comparison with MIOpen Direct API

The MIOpen direct API (`test_miopen_conv.cc`) successfully handles both cases:
- **Conv without bias:** Uses `miopenConvolutionForward`
- **Conv with bias:** Uses `miopenConvolutionForward` followed by `miopenOpTensor` for bias addition

---

## Recommendations

1. **For production use:** Use MIOpen direct API for conv+bias operations
2. **For hipDNN:** Only use for standalone convolution operations until fusion support is added
3. **Alternative:** Investigate hipDNN's IREE/Fusilli backend which may support more fusion patterns

---

## Raw Test Output

```
Running main() from googletest/src/gtest_main.cc
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from HipDNNConvTest
[ RUN      ] HipDNNConvTest.ConvolutionWithoutBias
Building hipDNN graph for convolution...
Validating graph...
Building operation graph...
Creating execution plans...
Building plans...
Allocated workspace: 67108864 bytes
Executing hipDNN convolution...
Max difference (conv without bias): 1.19209e-06
[       OK ] HipDNNConvTest.ConvolutionWithoutBias (657 ms)
[ RUN      ] HipDNNConvTest.ConvolutionWithBias
Building hipDNN graph for conv + bias...
Validating graph...
Building operation graph...
Creating execution plans...
test_hipdnn_conv.cc(448): error: Failed
hipDNN create_execution_plans failed: No engine configurations available for the graph.

[  FAILED  ] HipDNNConvTest.ConvolutionWithBias (14 ms)
[----------] 2 tests from HipDNNConvTest (671 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (671 ms total)
[  PASSED  ] 1 test.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] HipDNNConvTest.ConvolutionWithBias

 1 FAILED TEST
```
