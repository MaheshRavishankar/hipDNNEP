# Claude Code Instructions for hipDNN EP

Note: Codex is now the primary repo-local automation path. The Codex source of
truth is `AGENTS.md`, with reusable Codex skills under `plugins/hipdnnep-codex/`.
This file remains for Claude compatibility.

## Project Overview

This is an out-of-tree ONNXRuntime Execution Provider (EP) that uses AMD's hipDNN library for accelerated inference on AMD GPUs. The EP is built as a plugin that can be dynamically loaded by ONNXRuntime.

### Supported Operations

- **Conv2D** - via hipDNN graph API
- **MatMul/Gemm** - via hipBLAS-LT (optional, requires hipblaslt)

### Optional Features

- **hipBLAS-LT support** - Automatically enabled when hipblaslt is found. Provides optimized MatMul/Gemm operations.
- **Torch-MLIR integration** - Experimental IR-based compilation pipeline. Enable with `HIPDNN_EP_ENABLE_TORCH_MLIR=ON`.

## Build Commands

### Standard Build (without Torch-MLIR)

```bash
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

Tests use Google Test framework.

```bash
ctest --preset RelWithDebInfo --output-on-failure

# Or with MLIR build
ctest --preset RelWithDebInfo-MLIR --output-on-failure
```

### Lit Tests (Torch-MLIR)

Torch-MLIR IR generation tests use LLVM's lit framework with FileCheck. These are
automatically included in the test presets when Torch-MLIR is enabled.

#### FileCheck Rules

When writing CHECK lines, follow these conventions:

1. **Align colons**: Pad CHECK directives so all `:` align vertically
   ```
   CHECK-LABEL: func_name
         CHECK: module {
    CHECK-SAME:   arg1
   ```

2. **Capture SSA variables**: Don't hardcode SSA names like `%0`. Capture with `%[[NAME:.*]]` and reference as `%[[NAME]]`. Keep `%` outside the capture:
   ```
   CHECK: %[[RESULT:.*]] = torch.operator
   CHECK: return %[[RESULT]]
   ```

3. **Use CHECK-SAME for arguments**: Split long lines across multiple CHECK-SAME:
   ```
         CHECK: func.func @main
    CHECK-SAME:   (%[[A:.*]]: !torch.vtensor<[2,3],f32>,
    CHECK-SAME:    %[[B:.*]]: !torch.vtensor<[3,4],f32>)
    CHECK-SAME:   -> !torch.vtensor<[2,4],f32>
   ```

## Agent Sandbox

Implementer agents MUST always run inside the bwrap sandbox via
`scripts/hipdnn-sandbox.sh`. Never use the Claude Code Agent tool to
launch implementer agents directly — it lacks the correct filesystem
isolation (worktrees are outside the main checkout and need read-write
access). The sandbox script mounts the worktree read-write, the main
checkout read-only, and provides access to SDKs and build directories.

```bash
# Launch an implementer for a bead:
scripts/hipdnn-sandbox.sh <bead-id> -- -p "<prompt>" --max-turns 100
```

The PM skill (`/pm`) handles this automatically. See
`.claude/agents/implementer.md` for the full agent lifecycle.

## Notes

- Conv2D via hipDNN graph API
- MatMul/Gemm via hipBLAS-LT (when available)
- Plugin EP v2 API for dynamic loading
- Python tests require `onnx` package (install in `.venv` or system Python)

## Torch-MLIR Integration (Experimental)

The project includes optional Torch-MLIR integration for an IR-based compilation
pipeline. When enabled (`HIPDNN_EP_ENABLE_TORCH_MLIR=ON`), the EP converts ONNX
subgraphs to Torch-MLIR IR, runs an offload pipeline that compiles supported ops
into hipDNN executables (via the hipDNN/Fusilli backend), and lowers the result to builtin
tensor types ready for bufferization. The pipeline is defined in
`buildOffloadPipeline` (`passes.cc`).

See `docs/TorchMLIRAsIR/` for design documents and implementation plans.

### Session Config Options

These options are set via `OrtSessionOptions::AddConfigEntry`:

| Key | Default | Description |
|-----|---------|-------------|
| `hipdnn.use_torch_mlir` | `"0"` | Enable the Torch-MLIR compilation path |
| `hipdnn.dump_input_module` | `"0"` | Print MLIR to stdout after building the input module (before the offload pipeline) |
| `hipdnn.dump_lowered_module` | `"0"` | Print MLIR to stdout after running the offload pipeline |

### Debugging with `hipdnn-ep-opt`

The `hipdnn-ep-opt` tool (built with the MLIR preset) is an `mlir-opt`-like tool
that registers all hipDNN passes and the `--hipdnn-offload-pipeline` named pipeline.
This allows debugging the pipeline with standard MLIR flags.

**Workflow: Inspect IR after each pass**

1. Capture the input module by running with `hipdnn.dump_input_module` set to `"1"`:
   ```cpp
   session_options.AddConfigEntry("hipdnn.dump_input_module", "1");
   ```
   Save the output to a file (e.g., `input_module.mlir`).

2. Run the standalone tool with IR printing flags:
   ```bash
   hipdnn-ep-opt \
     --hipdnn-offload-pipeline \
     --mlir-print-ir-after-all \
     --mlir-print-local-scope \
     --mlir-disable-threading \
     input_module.mlir
   ```
   This prints the IR after each of the 4 pipeline passes, showing how the module
   transforms at each step.

3. To run individual passes instead of the full pipeline:
   ```bash
   # Just the offload pass (requires torch dialect input)
   hipdnn-ep-opt --hipdnn-offload input_module.mlir

   # Just the backend legalize pass (requires post-graph-to-executable input)
   hipdnn-ep-opt --hipdnn-backend-legalize post_executable.mlir
   ```

### Torch-MLIR Submodule

The torch-mlir source is included as a git submodule at `third_party/torch-mlir`.
It includes LLVM/MLIR as a nested dependency. To initialize:

```bash
git submodule update --init --recursive
```
