# Claude Code Instructions for hipDNN EP

## Project Overview

This is an out-of-tree ONNXRuntime Execution Provider (EP) that uses AMD's hipDNN library for accelerated inference on AMD GPUs. The EP is built as a plugin that can be dynamically loaded by ONNXRuntime.

## Build Commands

```bash
# Configure
cmake --preset RelWithDebInfo

# Build
cmake --build --preset RelWithDebInfo

# Test
ctest --preset RelWithDebInfo
```

## Environment Setup

Before building, ensure these environment variables are set:
```bash
export HIPDNN_ROOT="/path/to/hipDNN/stage"
export ONNXRUNTIME_ROOT="/path/to/onnxruntime"
```

## Project Structure

- `include/hipdnn_ep/` - Public headers
- `src/` - Implementation files
- `src/kernels/` - Kernel implementations (Conv, etc.)
- `test/` - Unit tests
- `cmake/` - CMake modules
- `docs/InitialBringUp/` - Design documents

## Key Components

- **HipDNNEpFactory** (`ep_factory.h/cc`) - Factory for creating EP instances, device discovery
- **HipDNNEp** (`ep.h/cc`) - Main EP class, graph partitioning, kernel compilation
- **HipDeviceAllocator** (`ep_allocator.h/cc`) - GPU memory allocator
- **HipDataTransfer** (`ep_data_transfer.h/cc`) - CPU<->GPU data transfers
- **ConvKernel** (`kernels/conv_kernel.h/cc`) - Convolution implementation using hipDNN

## Code Style

- Use `static` functions in anonymous namespaces for file-local helpers
- Keep implementation details out of headers where possible
- Use RAII and smart pointers for resource management
- Add TODOs for known limitations or future work

## Testing

Tests use Google Test framework. Run with:
```bash
ctest --preset RelWithDebInfo --output-on-failure
```

## Notes

- Currently supports Conv2D operations only
- Uses hipDNN graph API (not legacy immediate API)
- Plugin EP v2 API for dynamic loading
