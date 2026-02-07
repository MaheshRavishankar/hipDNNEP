# hipDNN Execution Provider for ONNXRuntime

An out-of-tree Execution Provider for ONNXRuntime that uses AMD's hipDNN library for accelerated inference on AMD GPUs.

## Status

**Work in Progress** - This is a prototype implementation.

### Supported Operations

- **Conv2D** - via hipDNN graph API
- **MatMul/Gemm** - via hipDNN graph API

### Optional Features

- **hipBLAS-LT support** - Automatically enabled when hipblaslt is found
- **Torch-MLIR integration** - Experimental IR-based compilation pipeline

## Tested Dependency Versions

| Dependency | Commit |
|------------|--------|
| [TheRock](https://github.com/ROCm/TheRock) | [`9639502b`](https://github.com/ROCm/TheRock/commit/9639502b523fff3faa7435894c61b1022ada9577) |
| [IREE](https://github.com/iree-org/iree) | [`db9d11e4`](https://github.com/iree-org/iree/commit/db9d11e4c000f693e9a70f7d3100db0c5294db9e) |

## Prerequisites

- CMake 3.20+
- Ninja build system
- HIP SDK (from TheRock)
- hipDNN library (from TheRock)
- hipBLAS-LT (optional, from TheRock) - enables MatMul/Gemm support
- ONNXRuntime (source and built library)
- iree-compile (required by hipDNN backend for code generation)
- Python 3 with `onnx` package (for test model generation)

## Building

### 1. Set Environment Variables

```bash
export THEROCK_DIST="/path/to/TheRock/build/dist/rocm"
export ONNXRUNTIME_ROOT="/path/to/onnxruntime"
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

Tests require `iree-compile` in PATH. The recommended approach is to create local
test presets in `CMakeUserPresets.json` (git-ignored) that set up the environment.

Example `CMakeUserPresets.json`:
```json
{
  "version": 4,
  "testPresets": [
    {
      "name": "RelWithDebInfo-local",
      "inherits": "RelWithDebInfo",
      "environment": {
        "PATH": "/path/to/iree/build/tools:$penv{PATH}"
      }
    }
  ]
}
```

Then run tests with the local preset:
```bash
ctest --preset RelWithDebInfo-local
```

Alternatively, set PATH manually before running tests:
```bash
export PATH="/path/to/iree/build/tools:$PATH"
ctest --preset RelWithDebInfo
```

### 4. Build with Torch-MLIR (Optional)

For the experimental IR-based compilation pipeline:

```bash
# First build torch-mlir (one-time setup, see CLAUDE.md for details)
# Then:
cmake --preset RelWithDebInfo-MLIR
cmake --build --preset RelWithDebInfo-MLIR
ctest --preset RelWithDebInfo-MLIR-local
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
3. **Kernel** (`Kernel`): Routes to appropriate backend (hipDNN, hipBLAS-LT, or Torch-MLIR)
4. **HipDNNGraph**: Builds hipDNN graph from ONNX nodes for Conv2D
5. **BlasGraph**: Builds hipBLAS-LT operations for MatMul/Gemm (when available)
6. **IRBuilder**: Torch-MLIR IR generation (experimental, when enabled)
7. **NodeComputeInfo**: ORT callback interface for kernel lifecycle
8. **Allocator** (`HipDeviceAllocator`): HIP device memory allocation
9. **Data Transfer** (`HipDataTransfer`): CPU <-> GPU data copies

### Backend Selection

The `Kernel` class automatically selects the appropriate backend:

1. **Torch-MLIR path** (if enabled): Converts ONNX to Torch-MLIR IR for compilation
2. **hipBLAS-LT** (if available): Used for MatMul/Gemm operations
3. **hipDNN graph API**: Used for Conv2D and other supported operations

### hipDNN Integration

hipDNN uses a graph-based execution model:

1. Build operation graph from ONNX nodes (conv_fprop, etc.)
2. Validate and create execution plans
3. Execute with variant pack (tensor uid -> device pointer mapping)

## License

MIT License
