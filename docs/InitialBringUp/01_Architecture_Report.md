# hipDNN EP Architecture Report

## 1. Executive Summary

This document provides a comprehensive analysis of the architecture needed to build an out-of-tree Execution Provider (EP) for ONNXRuntime that uses hipDNN as its backend. The EP will initially support convolution operations, with plans to extend to other operations and fusion patterns.

---

## 2. ONNXRuntime EP Architecture

### 2.1 Core Interfaces

ONNXRuntime provides two approaches for building EPs:

1. **In-tree EPs** - Built as part of ONNXRuntime (CUDA, ROCm, CPU, etc.)
2. **Plugin EPs** - Out-of-tree EPs loaded dynamically at runtime

For this project, we'll use the **Plugin EP** approach, which allows:
- Independent development and versioning
- No modifications to ONNXRuntime source
- Dynamic registration at runtime

### 2.2 Plugin EP Interface

Location: `onnxruntime/core/session/plugin_ep/`

**Required Exports:**

```c
// Create EP factories (entry point)
OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories);

// Release EP factory
OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

### 2.3 OrtEpFactory Interface

The factory creates EP instances and manages device discovery:

```c
struct OrtEpFactory {
    // Metadata
    const char* GetName();
    const char* GetVendor();
    uint32_t GetVendorId();
    const char* GetVersion();

    // Device discovery
    OrtStatus* GetSupportedDevices(
        const OrtHardwareDevice* const* devices,
        size_t num_devices,
        OrtEpDevice** ep_devices,
        size_t max_ep_devices,
        size_t* num_ep_devices);

    // EP lifecycle
    OrtStatus* CreateEp(...);
    void ReleaseEp(OrtEp* ep);

    // Memory management
    OrtStatus* CreateAllocator(...);
    void ReleaseAllocator(OrtAllocator* allocator);
    OrtStatus* CreateDataTransfer(OrtDataTransferImpl** data_transfer);
};
```

### 2.4 Key Components to Implement

| Component | Purpose |
|-----------|---------|
| `OrtEpFactory` | Factory for creating EP instances |
| `OrtEp` | Main EP implementation |
| `OrtAllocator` | Device memory allocator |
| `OrtDataTransferImpl` | CPU ↔ Device data transfer |
| Kernel Registry | Maps ONNX ops to hipDNN operations |

---

## 3. hipDNN Library Overview

### 3.1 Architecture

hipDNN is a graph-based deep learning library with:
- **Backend** (`hipdnn_backend`) - C API for graph execution
- **Frontend** (`hipdnn_frontend`) - Header-only C++ wrapper
- **Plugin System** - Swappable execution engines (MIOpen, Fusilli/IREE)

### 3.2 Execution Model

hipDNN uses a graph-based execution model:

```
1. Create handle
2. Build operation graph (using frontend API)
3. Validate graph
4. Create execution plans (with heuristics)
5. Allocate workspace
6. Execute with variant pack (uid → device_ptr mapping)
```

### 3.3 Convolution API

```cpp
// Forward convolution
auto y = graph->conv_fprop(x, w, ConvFpropAttributes()
    .set_padding({padH, padW})
    .set_stride({strideH, strideW})
    .set_dilation({dilH, dilW}));

// Backward data
auto dx = graph->conv_dgrad(dy, w, ConvDgradAttributes());

// Backward filter
auto dw = graph->conv_wgrad(dy, x, ConvWgradAttributes());
```

### 3.4 Supported Data Types

- `DataType::FLOAT` (fp32)
- `DataType::HALF` (fp16)
- `DataType::BFLOAT16` (bf16)

### 3.5 Key Files

| Path | Description |
|------|-------------|
| `rocm-libraries/projects/hipdnn/frontend/include/hipdnn_frontend.hpp` | Main C++ header |
| `rocm-libraries/projects/hipdnn/backend/include/hipdnn_backend.h` | C backend API |
| `rocm-libraries/projects/hipdnn/samples/ConvFprop.cpp` | Conv example |

---

## 4. Mapping ONNX to hipDNN

### 4.1 Conv Operation

| ONNX Attribute | hipDNN Attribute |
|----------------|------------------|
| `kernel_shape` | Implicit from weight tensor |
| `pads` | `set_padding()` |
| `strides` | `set_stride()` |
| `dilations` | `set_dilation()` |
| `group` | TBD (may need special handling) |
| `auto_pad` | Compute padding manually |

### 4.2 Data Layout

- ONNX default: NCHW
- hipDNN: NCHW (compatible)

### 4.3 Type Mapping

| ONNX Type | hipDNN Type |
|-----------|-------------|
| `float` | `DataType::FLOAT` |
| `float16` | `DataType::HALF` |
| `bfloat16` | `DataType::BFLOAT16` |

---

## 5. Reference Implementation

ONNXRuntime provides example plugin EPs in:
```
onnxruntime/test/autoep/library/
├── example_plugin_ep/          # Basic plugin EP
├── example_plugin_ep_virt_gpu/ # Virtual GPU device
└── example_plugin_ep_kernel_registry/  # Kernel-based approach
```

Key files to study:
- `ep_factory.h/cc` - Factory implementation
- `ep.h/cc` - EP implementation
- `ep_allocator.h` - Memory allocator
- `ep_data_transfer.h/cc` - Data transfer
- `example_plugin_ep.cc` - Export functions

---

## 6. Build System

### 6.1 Dependencies

- ONNXRuntime headers (C API)
- hipDNN (backend + frontend)
- HIP runtime
- CMake 3.20+

### 6.2 CMake Structure

```cmake
find_package(hip REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend CONFIG REQUIRED)

add_library(hipdnn_ep SHARED
    src/ep_factory.cc
    src/ep.cc
    src/ep_allocator.cc
    src/ep_data_transfer.cc
    src/kernels/conv_kernel.cc
    src/hipdnn_ep.cc  # exports
)

target_link_libraries(hipdnn_ep
    hipdnn_frontend
    hipdnn_backend
    hip::host
)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

- Individual kernel correctness
- Memory allocation/deallocation
- Data transfer CPU ↔ GPU

### 7.2 Integration Tests

- Load EP via `RegisterExecutionProviderLibrary()`
- Run inference with Conv models
- Compare outputs with CPU EP

### 7.3 ONNX Model Tests

- Single Conv layer models
- Multi-layer CNN models
- Models with different data types

---

## 8. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| EP Type | Plugin EP | Out-of-tree, independent development |
| Approach | Kernel Registry | More flexible, supports partial graph execution |
| Initial Op | Conv | Most common, good learning example |
| Data Layout | NCHW | Match ONNX default |

---

## 9. Files to Reference

### ONNXRuntime

| File | Purpose |
|------|---------|
| `include/onnxruntime/core/session/onnxruntime_c_api.h` | Main C API |
| `include/onnxruntime/core/session/onnxruntime_ep_c_api.h` | EP C API |
| `include/onnxruntime/core/session/onnxruntime_cxx_api.h` | C++ wrapper |
| `core/session/plugin_ep/ep_library_plugin.h` | Plugin loading |
| `test/autoep/library/example_plugin_ep/` | Example EP |

### hipDNN

| File | Purpose |
|------|---------|
| `hipdnn_frontend.hpp` | Main header |
| `Graph.hpp` | Graph building |
| `ConvolutionFpropAttributes.hpp` | Conv attributes |
| `samples/ConvFprop.cpp` | Usage example |

---

## 10. Next Steps

See `02_Execution_Plan.md` for the detailed implementation plan.
