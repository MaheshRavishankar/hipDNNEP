# Claude Code Instructions for hipDNN EP

## Project Overview

This is an out-of-tree ONNXRuntime Execution Provider (EP) that uses AMD's hipDNN library for accelerated inference on AMD GPUs. The EP is built as a plugin that can be dynamically loaded by ONNXRuntime.

## Build Commands

```bash
# Ensure iree-compile is in PATH
export PATH="/path/to/iree/build/tools:$PATH"

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
export THEROCK_DIST="/path/to/TheRock/build/dist/rocm"
export ONNXRUNTIME_ROOT="/path/to/onnxruntime"
export PATH="/path/to/iree/build/tools:$PATH"
```

## Git Workflow

- **Branch naming**: `users/<author>/<branchName>` (camelCase)
  - Example: `users/MaheshRavishankar/addClangFormat`
- **Main branch**: Protected, requires PR with 1 approval
- **Merge method**: Squash only

## Key Components

- **HipDNNEpFactory** (`ep_factory.h/cc`) - Factory for creating EP instances, device discovery
- **HipDNNEp** (`ep.h/cc`) - Main EP class, graph partitioning, kernel compilation
- **Kernel** (`kernel.h/cc`) - Builds hipDNN graph from ONNX nodes, executes inference
- **NodeComputeInfo** (`node_compute_info.h/cc`) - ORT callback interface for kernel lifecycle
- **HipDeviceAllocator** (`ep_allocator.h/cc`) - GPU memory allocator
- **HipDataTransfer** (`ep_data_transfer.h/cc`) - CPU<->GPU data transfers

## Code Style

- Uses clang-format with Google style base (see `.clang-format`)
- Use `static` functions in anonymous namespaces for file-local helpers
- Keep implementation details out of headers where possible
- Use RAII and smart pointers for resource management

## Testing

Tests use Google Test framework. Run with:
```bash
ctest --preset RelWithDebInfo --output-on-failure
```

## Notes

- Currently supports Conv2D operations only
- Uses hipDNN graph API (not legacy immediate API)
- Plugin EP v2 API for dynamic loading
- Requires iree-compile in PATH for hipDNN backend code generation
