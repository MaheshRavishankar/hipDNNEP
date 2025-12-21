# hipDNN Execution Provider for ONNXRuntime

An out-of-tree Execution Provider for ONNXRuntime that uses AMD's hipDNN library for accelerated inference on AMD GPUs.

## Status

**Work in Progress** - This is a prototype implementation.

Currently supported operations:
- Conv (2D convolution)

## Prerequisites

- CMake 3.20+
- Ninja build system
- HIP SDK (from TheRock)
- hipDNN library (from TheRock)
- ONNXRuntime (source and built library)
- iree-compile (required by hipDNN backend for code generation)

## Building

### 1. Set Environment Variables

```bash
export THEROCK_DIST="/path/to/TheRock/build/dist/rocm"
export ONNXRUNTIME_ROOT="/path/to/onnxruntime"

# iree-compile must be in PATH
export PATH="/path/to/iree/build/tools:$PATH"
```

### 2. Configure and Build

```bash
cd hipDNNEP

# Configure
cmake --preset RelWithDebInfo

# Build
cmake --build --preset RelWithDebInfo
```

### 3. Run Tests

```bash
ctest --preset RelWithDebInfo
```

## Usage

### Loading the EP in ONNXRuntime

```cpp
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::InitApi(OrtGetApiBase()->GetApi(ORT_API_VERSION));
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example");

    // Register the hipDNN EP library
    OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
        env, "HipDNN", "/path/to/libhipdnn_ep.so");

    if (status != nullptr) {
        // Handle error
        Ort::GetApi().ReleaseStatus(status);
        return 1;
    }

    // Get available EP devices
    std::vector<Ort::ConstEpDevice> devices = env.GetEpDevices();

    // Find HipDNN device
    const OrtEpDevice* hipdnn_device = nullptr;
    for (const auto& device : devices) {
        if (device.EpName() == "HipDNN") {
            hipdnn_device = static_cast<const OrtEpDevice*>(device);
            break;
        }
    }

    // Create session options and append EP
    Ort::SessionOptions session_options;
    Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        session_options, env, &hipdnn_device, 1, nullptr, nullptr, 0);

    // Create session
    Ort::Session session(env, "model.onnx", session_options);

    // Run inference
    // ...

    return 0;
}
```

## Architecture

This EP uses the ONNXRuntime Plugin EP V2 system, which allows:

- Building as a separate shared library
- Dynamic loading at runtime
- No modifications to ONNXRuntime source

### Key Components

1. **EP Factory** (`HipDNNEpFactory`): Creates EP instances and manages device discovery
2. **EP** (`HipDNNEp`): Main execution provider, handles graph partitioning and compilation
3. **Kernel** (`Kernel`): Builds hipDNN graph from ONNX nodes and executes inference
4. **NodeComputeInfo**: ORT callback interface for kernel lifecycle
5. **Allocator** (`HipDeviceAllocator`): HIP device memory allocation
6. **Data Transfer** (`HipDataTransfer`): CPU <-> GPU data copies

### hipDNN Integration

hipDNN uses a graph-based execution model:

1. Build operation graph from ONNX nodes (conv_fprop, etc.)
2. Validate and create execution plans
3. Execute with variant pack (tensor uid -> device pointer mapping)

The `Kernel` class maintains a symbol table mapping ONNX value names to hipDNN TensorAttributes,
enabling multi-node graph construction and future op fusion.

## License

MIT License
