# hipDNN EP Execution Plan

## Overview

This document outlines the step-by-step plan to build the hipDNN Execution Provider for ONNXRuntime.

---

## Phase 1: Project Setup

### 1.1 Repository Structure

Create the following structure in `/home/mahesh/onnxruntime/hipDNNEP`:

```
hipDNNEP/
├── CMakeLists.txt              # Root CMake file
├── cmake/
│   └── FindONNXRuntime.cmake   # Find module for ORT headers
├── include/
│   └── hipdnn_ep/
│       ├── ep_factory.h        # Factory interface
│       ├── ep.h                # EP implementation
│       ├── ep_allocator.h      # HIP memory allocator
│       ├── ep_data_transfer.h  # CPU ↔ GPU transfer
│       └── kernels/
│           └── conv_kernel.h   # Conv kernel wrapper
├── src/
│   ├── ep_factory.cc
│   ├── ep.cc
│   ├── ep_allocator.cc
│   ├── ep_data_transfer.cc
│   ├── hipdnn_ep_exports.cc    # DLL exports
│   └── kernels/
│       └── conv_kernel.cc
├── test/
│   ├── CMakeLists.txt
│   ├── test_conv.cc            # Conv kernel tests
│   ├── test_ep_load.cc         # EP loading tests
│   └── models/                 # Test ONNX models
└── README.md
```

### 1.2 CMake Configuration

**Tasks:**
- [ ] Create root CMakeLists.txt with find_package for HIP, hipDNN, ONNXRuntime headers
- [ ] Set up proper compile flags for HIP code
- [ ] Configure shared library output with proper symbol visibility
- [ ] Create CMake presets for Debug/RelWithDebInfo builds

### 1.3 Dependencies

**Required:**
- ONNXRuntime headers (from `/home/mahesh/onnxruntime/onnxruntime/include/`)
- hipDNN (from `/home/mahesh/TheRock/MaheshRelWithDebInfo`)
- HIP runtime

**CMake find_package targets:**
```cmake
find_package(hip REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED
    PATHS /home/mahesh/TheRock/MaheshRelWithDebInfo/...)
find_package(hipdnn_backend CONFIG REQUIRED
    PATHS /home/mahesh/TheRock/MaheshRelWithDebInfo/...)
```

---

## Phase 2: Core EP Infrastructure

### 2.1 EP Factory Implementation

**File:** `src/ep_factory.cc`

**Tasks:**
- [ ] Implement `HipDNNEpFactory` class inheriting OrtEpFactory pattern
- [ ] Implement `GetName()` returning "HipDNNExecutionProvider"
- [ ] Implement `GetVendor()` returning "AMD"
- [ ] Implement `GetSupportedDevices()` to detect AMD GPUs
- [ ] Implement `CreateEp()` to instantiate the EP
- [ ] Implement `CreateAllocator()` for HIP memory
- [ ] Implement `CreateDataTransfer()` for host-device copies

**Reference:** `onnxruntime/test/autoep/library/example_plugin_ep/ep_factory.cc`

### 2.2 Export Functions

**File:** `src/hipdnn_ep_exports.cc`

```cpp
extern "C" {
    EXPORT_SYMBOL OrtStatus* CreateEpFactories(
        const char* registration_name,
        const OrtApiBase* ort_api_base,
        const OrtLogger* default_logger,
        OrtEpFactory** factories,
        size_t max_factories,
        size_t* num_factories);

    EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(
        OrtEpFactory* factory);
}
```

### 2.3 Memory Allocator

**File:** `src/ep_allocator.cc`

**Tasks:**
- [ ] Implement HIP device memory allocation using `hipMalloc`/`hipFree`
- [ ] Track allocations for debugging
- [ ] Handle allocation failures gracefully

### 2.4 Data Transfer

**File:** `src/ep_data_transfer.cc`

**Tasks:**
- [ ] Implement `CanCopy()` for CPU ↔ HIP device
- [ ] Implement `CopyTensor()` using `hipMemcpy`
- [ ] Support async copies using streams

---

## Phase 3: EP Implementation

### 3.1 Main EP Class

**File:** `src/ep.cc`

**Tasks:**
- [ ] Implement `HipDNNExecutionProvider` class
- [ ] Maintain hipDNN handle per EP instance
- [ ] Implement `GetCapability()` to claim supported nodes
- [ ] Implement kernel lookup for Conv operations
- [ ] Handle node compilation with `Compile()`

### 3.2 GetCapability Implementation

Key logic:
```cpp
std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
    const GraphViewer& graph,
    const IKernelLookup& kernel_lookup) {

    std::vector<std::unique_ptr<ComputeCapability>> capabilities;

    for (const Node& node : graph.Nodes()) {
        if (IsSupportedOp(node)) {
            // Create capability for this node
            auto sub_graph = std::make_unique<IndexedSubGraph>();
            sub_graph->nodes.push_back(node.Index());

            capabilities.push_back(
                std::make_unique<ComputeCapability>(std::move(sub_graph)));
        }
    }
    return capabilities;
}
```

### 3.3 Supported Operation Check

```cpp
bool IsSupportedOp(const Node& node) {
    const std::string& op_type = node.OpType();

    if (op_type == "Conv") {
        // Check constraints:
        // - 2D convolution (4D tensors)
        // - Supported data types (float, float16)
        // - No unsupported attributes
        return CheckConvSupport(node);
    }
    return false;
}
```

---

## Phase 4: Conv Kernel Implementation

### 4.1 Conv Kernel Wrapper

**File:** `src/kernels/conv_kernel.cc`

**Tasks:**
- [ ] Parse ONNX Conv attributes (pads, strides, dilations, group)
- [ ] Create hipDNN graph for convolution
- [ ] Build execution plan
- [ ] Execute with input/output tensors

### 4.2 ONNX to hipDNN Mapping

```cpp
class ConvKernel {
public:
    ConvKernel(const Node& node, hipdnnHandle_t handle);

    Status Compute(
        const Tensor& X,      // Input
        const Tensor& W,      // Weight
        const Tensor* B,      // Optional bias
        Tensor& Y);           // Output

private:
    std::shared_ptr<hipdnn::graph::Graph> graph_;
    hipdnnHandle_t handle_;

    // Cached attributes
    std::vector<int64_t> pads_;
    std::vector<int64_t> strides_;
    std::vector<int64_t> dilations_;
    int64_t group_;
};
```

### 4.3 Graph Building

```cpp
Status ConvKernel::BuildGraph(const TensorShape& x_shape,
                               const TensorShape& w_shape) {
    graph_ = std::make_shared<hipdnn::graph::Graph>();
    graph_->set_io_data_type(DataType::FLOAT)
           .set_compute_data_type(DataType::FLOAT);

    // Create tensor attributes
    auto x_attr = CreateTensorAttr("X", x_shape, /*uid=*/1);
    auto w_attr = CreateTensorAttr("W", w_shape, /*uid=*/2);

    // Convolution attributes
    ConvFpropAttributes conv_attrs;
    conv_attrs.set_padding({pads_[0], pads_[1]})
              .set_stride({strides_[0], strides_[1]})
              .set_dilation({dilations_[0], dilations_[1]});

    // Add convolution
    auto y_attr = graph_->conv_fprop(x_attr, w_attr, conv_attrs);
    y_attr->set_uid(3);

    // Validate and build
    graph_->validate();
    graph_->build_operation_graph(handle_);
    graph_->create_execution_plans({HeuristicMode::FALLBACK});
    graph_->check_support();
    graph_->build_plans();

    return Status::OK();
}
```

---

## Phase 5: Testing

### 5.1 Unit Tests

**File:** `test/test_conv.cc`

**Tests:**
- [ ] Basic 2D convolution correctness
- [ ] Different kernel sizes (1x1, 3x3, 5x5, 7x7)
- [ ] Strided convolution
- [ ] Padded convolution
- [ ] Dilated convolution
- [ ] Different batch sizes
- [ ] Different channel counts

### 5.2 EP Loading Tests

**File:** `test/test_ep_load.cc`

**Tests:**
- [ ] Load EP library successfully
- [ ] Register with session
- [ ] Run simple inference

### 5.3 Test Infrastructure

```cpp
// Helper to run ONNX model and compare outputs
Status RunAndCompare(const std::string& model_path,
                     const std::vector<Tensor>& inputs,
                     float tolerance = 1e-5f) {
    // Run with CPU EP
    auto cpu_outputs = RunWithCpuEp(model_path, inputs);

    // Run with hipDNN EP
    auto hipdnn_outputs = RunWithHipDNNEp(model_path, inputs);

    // Compare
    return CompareOutputs(cpu_outputs, hipdnn_outputs, tolerance);
}
```

---

## Phase 6: Integration & Documentation

### 6.1 Integration with ONNXRuntime

**Usage example:**
```cpp
Ort::Env env;
Ort::SessionOptions session_options;

// Register hipDNN EP
Ort::ThrowOnError(Ort::GetApi().RegisterExecutionProviderLibrary(
    env, "HipDNN", "/path/to/libhipdnn_ep.so"));

// Get devices
auto ep_devices = Ort::GetEpDevices(env);

// Append to session
Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
    session_options, env, ep_devices.data(), ep_devices.size(), ...));

// Create session
Ort::Session session(env, "model.onnx", session_options);
```

### 6.2 Documentation

- [ ] README with build instructions
- [ ] API documentation
- [ ] Supported operations list
- [ ] Performance tuning guide

---

## Phase 7: Future Extensions

### 7.1 Additional Operations

| Priority | Operation | hipDNN Support |
|----------|-----------|----------------|
| High | BatchNormalization | Yes |
| High | Relu/Activation | Yes (Pointwise) |
| Medium | MaxPool/AvgPool | TBD |
| Medium | Gemm/MatMul | TBD |
| Low | Other ops | As needed |

### 7.2 Fusion Support

hipDNN supports fused operations:
- Conv + Bias + ReLU
- Conv + BatchNorm
- BatchNorm + ReLU

**Implementation approach:**
1. Implement pattern matching in `GetCapability()`
2. Claim multiple nodes as fused subgraph
3. Build single hipDNN graph for fused operations

### 7.3 Performance Optimizations

- [ ] Kernel caching (avoid rebuilding graphs)
- [ ] Workspace memory pooling
- [ ] Stream-based async execution
- [ ] Auto-tuning support

---

## Implementation Order

### Week 1-2: Foundation
1. Project setup and CMake configuration
2. EP factory and exports
3. Memory allocator
4. Data transfer

### Week 3-4: Core EP
5. EP implementation with GetCapability
6. Basic Conv kernel (no special cases)
7. Unit tests for Conv

### Week 5-6: Refinement
8. Handle Conv edge cases (groups, asymmetric padding)
9. Integration tests
10. Documentation

### Future
11. Additional operations (BatchNorm, Relu)
12. Fusion patterns
13. Performance optimization

---

## Success Criteria

### Phase 1 Complete
- [ ] Project builds successfully
- [ ] EP library loads without errors

### Phase 2 Complete
- [ ] EP registers with ONNXRuntime
- [ ] Device detection works

### Phase 3 Complete
- [ ] Conv nodes are claimed by EP
- [ ] Basic inference runs

### Phase 4 Complete
- [ ] Conv outputs match CPU EP within tolerance
- [ ] Multiple Conv configurations work

### Phase 5 Complete
- [ ] All unit tests pass
- [ ] Integration tests pass

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| hipDNN API instability | Pin to specific TheRock version |
| ONNXRuntime API changes | Target specific ORT version |
| Performance issues | Start simple, optimize later |
| Missing hipDNN features | Fall back to CPU for unsupported ops |

---

## Resources

- ONNXRuntime Plugin EP Docs: https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries.html
- hipDNN samples: `/home/mahesh/TheRock/TheRock/rocm-libraries/projects/hipdnn/samples/`
- Example Plugin EP: `/home/mahesh/onnxruntime/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/`
