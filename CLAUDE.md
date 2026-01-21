# Claude Code Instructions for hipDNN EP

## Project Overview

This is an out-of-tree ONNXRuntime Execution Provider (EP) that uses AMD's hipDNN library for accelerated inference on AMD GPUs. The EP is built as a plugin that can be dynamically loaded by ONNXRuntime.

## Build Commands

### Standard Build (without Torch-MLIR)

```bash
# Ensure iree-compile is in PATH
export PATH="$HOME/iree/build/RelWithDebInfo/tools:$PATH"

# Configure
cmake --preset RelWithDebInfo

# Build
cmake --build --preset RelWithDebInfo

# Test
ctest --preset RelWithDebInfo
```

### Build with Torch-MLIR (for IR pipeline)

First, build torch-mlir and LLVM (one-time setup):

```bash
# Clone and build torch-mlir with LLVM
cd third_party/torch-mlir

# Create build directory
cmake -G Ninja -B ../../../build/torch-mlir \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=../../../build/torch-mlir/install \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
  -DTORCH_MLIR_ENABLE_REFBACKEND=OFF \
  -DTORCH_MLIR_USE_INSTALLED_PYTORCH=OFF \
  -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF \
  externals/llvm-project/llvm

# Build and install (this takes a while)
cmake --build ../../../build/torch-mlir --target install

cd ../..
```

Then build hipDNNEP with Torch-MLIR:

```bash
# Configure with MLIR support
cmake --preset RelWithDebInfo-MLIR

# Build
cmake --build --preset RelWithDebInfo-MLIR

# Test
ctest --preset RelWithDebInfo-MLIR
```

## Environment Setup

Before building, ensure these environment variables are set:
```bash
export THEROCK_DIST="/path/to/TheRock/build/dist/rocm"
export ONNXRUNTIME_ROOT="/path/to/onnxruntime"
export PATH="$HOME/iree/build/RelWithDebInfo/tools:$PATH"
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

- Currently supports Conv2D, MatMul, and Gemm operations
- Uses hipDNN graph API (not legacy immediate API)
- Plugin EP v2 API for dynamic loading
- Requires iree-compile in PATH for hipDNN backend code generation

## Torch-MLIR Integration (Experimental)

The project includes optional Torch-MLIR integration for an IR-based compilation
pipeline. When enabled (`HIPDNN_EP_ENABLE_TORCH_MLIR=ON`), the EP can:

- Convert ONNX subgraphs to Torch-MLIR IR
- Apply pattern matching for hipDNN offload
- Use MLIR bufferization for memory planning

See `docs/TorchMLIRAsIR/` for design documents and implementation plans.

### Torch-MLIR Submodule

The torch-mlir source is included as a git submodule at `third_party/torch-mlir`.
It includes LLVM/MLIR as a nested dependency. To initialize:

```bash
git submodule update --init --recursive
```
