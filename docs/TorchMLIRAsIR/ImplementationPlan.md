# Implementation Plan: Torch-MLIR Integration for hipDNN EP

## Overview

This document outlines the implementation plan for integrating Torch-MLIR as an
intermediate representation for the hipDNN Execution Provider. The plan is
organized into phases, with each phase building on the previous one.

## Project Structure After Implementation

```
hipDNNEP/
├── CMakeLists.txt                    # Updated for torch-mlir
├── third_party/
│   └── torch-mlir/                   # Git submodule
│       └── externals/
│           └── llvm-project/         # Managed by torch-mlir
├── include/hipdnn_ep/
│   ├── kernel.h                      # Existing
│   ├── ir_builder.h                  # NEW: ORT to Torch-MLIR
│   ├── hipdnn_patterns.h             # NEW: Pattern matching
│   ├── hipdnn_lowering.h             # NEW: Lowering passes
│   └── executor.h                    # NEW: Runtime execution
├── src/
│   ├── kernel.cc                     # Modified: use IR pipeline
│   ├── ir_builder.cc                 # NEW: Build Torch-MLIR from ORT
│   ├── hipdnn_patterns.cc            # NEW: hipdnn.graph patterns
│   ├── hipdnn_lowering.cc            # NEW: Torch to memref lowering
│   └── executor.cc                   # NEW: Execute compiled IR
└── lib/Dialect/HipDNN/              # NEW: Optional custom dialect
    ├── IR/
    │   ├── HipDNNOps.td
    │   └── HipDNNOps.cpp
    └── Transforms/
        └── Passes.cpp
```

---

## Phase 1: Build Infrastructure

**Goal**: Set up torch-mlir as a submodule and integrate with CMake build.

### 1.1 Add Git Submodules

```bash
# Add torch-mlir submodule
git submodule add https://github.com/llvm/torch-mlir.git third_party/torch-mlir

# torch-mlir manages LLVM internally; no separate LLVM submodule needed
# It uses CMake FetchContent or external LLVM
```

### 1.2 Update CMakeLists.txt

```cmake
# Option to enable Torch-MLIR IR support
option(HIPDNN_EP_ENABLE_TORCH_MLIR "Enable Torch-MLIR IR pipeline" OFF)

if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  # torch-mlir options
  set(TORCH_MLIR_ENABLE_STABLEHLO OFF CACHE BOOL "" FORCE)
  set(TORCH_MLIR_ENABLE_REFBACKEND OFF CACHE BOOL "" FORCE)
  set(TORCH_MLIR_USE_INSTALLED_PYTORCH OFF CACHE BOOL "" FORCE)

  # Add torch-mlir (will build LLVM if not found)
  add_subdirectory(third_party/torch-mlir EXCLUDE_FROM_ALL)

  target_link_libraries(hipdnn_ep PRIVATE
    TorchMLIRTorchDialect
    TorchMLIRTorchPasses
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRBufferization
    MLIRPass
  )

  target_compile_definitions(hipdnn_ep PRIVATE HIPDNN_EP_HAS_TORCH_MLIR)
endif()
```

### 1.3 Verify Build

- [ ] Create CMake preset for Torch-MLIR enabled build
- [ ] Verify torch-mlir builds with ROCm toolchain
- [ ] Add CI job for torch-mlir enabled build
- [ ] Document build process in README

### 1.4 Files Changed/Added

| File | Action | Description |
|------|--------|-------------|
| `.gitmodules` | Create | Add torch-mlir submodule |
| `CMakeLists.txt` | Modify | Add torch-mlir integration |
| `CMakePresets.json` | Modify | Add preset with MLIR enabled |
| `CLAUDE.md` | Modify | Update build instructions |

---

## Phase 2: IR Builder - ORT to Torch-MLIR

**Goal**: Convert ONNXRuntime subgraphs to Torch-MLIR operations.

### 2.1 IR Builder Interface

```cpp
// include/hipdnn_ep/ir_builder.h
namespace hipdnn_ep {

class IRBuilder {
public:
  IRBuilder(mlir::MLIRContext& ctx);

  // Build Torch-MLIR function from ORT subgraph
  mlir::OwningOpRef<mlir::ModuleOp> BuildModule(
      const std::vector<Ort::ConstValueInfo>& inputs,
      const std::vector<Ort::ConstValueInfo>& outputs,
      const std::vector<Ort::ConstNode>& nodes);

private:
  mlir::Type ConvertType(ONNXTensorElementDataType dtype,
                         const std::vector<int64_t>& shape);

  mlir::Value BuildOperation(Ort::ConstNode node,
                             const llvm::DenseMap<std::string, mlir::Value>& symbolTable);

  // Operation-specific builders
  mlir::Value BuildConv(Ort::ConstNode node, mlir::ValueRange inputs);
  mlir::Value BuildMatMul(Ort::ConstNode node, mlir::ValueRange inputs);
  mlir::Value BuildGemm(Ort::ConstNode node, mlir::ValueRange inputs);
  mlir::Value BuildAdd(Ort::ConstNode node, mlir::ValueRange inputs);
  mlir::Value BuildRelu(Ort::ConstNode node, mlir::ValueRange inputs);

  mlir::MLIRContext& ctx_;
  mlir::OpBuilder builder_;
};

}  // namespace hipdnn_ep
```

### 2.2 Type Conversion

| ONNX Type | Torch-MLIR Type |
|-----------|-----------------|
| `float32` | `!torch.vtensor<[...],f32>` |
| `float16` | `!torch.vtensor<[...],f16>` |
| `int64`   | `!torch.vtensor<[...],si64>` |
| `int32`   | `!torch.vtensor<[...],si32>` |

### 2.3 Operation Mapping

| ONNX Op | Torch-MLIR Op | Notes |
|---------|---------------|-------|
| Conv | `torch.aten.conv2d` | NCHW layout |
| MatMul | `torch.aten.mm` | 2D only initially |
| Gemm | `torch.aten.addmm` | With optional bias |
| Add | `torch.aten.add.Tensor` | Elementwise |
| Relu | `torch.aten.relu` | Activation |

### 2.4 Files Changed/Added

| File | Action | Description |
|------|--------|-------------|
| `include/hipdnn_ep/ir_builder.h` | Create | IR builder header |
| `src/ir_builder.cc` | Create | IR builder implementation |
| `CMakeLists.txt` | Modify | Add new source file |

---

## Phase 3: hipDNN Pattern Matching

**Goal**: Identify operations/sequences that can be offloaded to hipDNN.

### 3.1 Pattern Design

Create patterns that:
1. Match supported operations (Conv, MatMul)
2. Optionally match fused patterns (Conv+Bias+Relu)
3. Outline matched regions into `torch.operator "hipdnn.graph"` ops

### 3.2 Pattern Interface

```cpp
// include/hipdnn_ep/hipdnn_patterns.h
namespace hipdnn_ep {

// Populate patterns for hipDNN offload
void populateHipDNNOffloadPatterns(mlir::RewritePatternSet& patterns,
                                    mlir::MLIRContext* ctx);

// Pass to apply hipDNN offload patterns
std::unique_ptr<mlir::Pass> createHipDNNOffloadPass();

}  // namespace hipdnn_ep
```

### 3.3 Supported Patterns

**Phase 3a: Single-op patterns**
- `torch.aten.conv2d` → `hipdnn.graph { conv2d }`
- `torch.aten.mm` → `hipdnn.graph { mm }`
- `torch.aten.addmm` → `hipdnn.graph { addmm }`

**Phase 3b: Fused patterns (future)**
- `conv2d + add + relu` → `hipdnn.graph { conv2d, add, relu }`
- `mm + add` → `hipdnn.graph { mm, add }`

### 3.4 Files Changed/Added

| File | Action | Description |
|------|--------|-------------|
| `include/hipdnn_ep/hipdnn_patterns.h` | Create | Pattern declarations |
| `src/hipdnn_patterns.cc` | Create | Pattern implementations |

---

## Phase 4: hipDNN Compilation

**Goal**: Compile `hipdnn.graph` regions to hipDNN graph API calls.

### 4.1 Compilation Flow

```
hipdnn.graph { torch ops }
        │
        ▼
    Extract region body
        │
        ▼
    Build hipDNN graph (existing HipDNNGraph code)
        │
        ▼
    Store in cache with unique name
        │
        ▼
    Replace hipdnn.graph with hipdnn.executable reference
```

### 4.2 Integration with Existing Code

Reuse `HipDNNGraph` class for compilation:
- Extract ops from Torch-MLIR region
- Convert back to ORT-like representation OR
- Modify HipDNNGraph to work directly with MLIR ops

**Recommendation**: Initially, convert MLIR ops to the same format that
`HipDNNGraph::Build` expects. This minimizes changes to working code.

### 4.3 Compiled Graph Cache

```cpp
class CompiledGraphCache {
public:
  void Store(llvm::StringRef name, std::unique_ptr<HipDNNGraph> graph);
  HipDNNGraph* Lookup(llvm::StringRef name);
  void Clear();

private:
  llvm::StringMap<std::unique_ptr<HipDNNGraph>> cache_;
};
```

### 4.4 Files Changed/Added

| File | Action | Description |
|------|--------|-------------|
| `include/hipdnn_ep/compiled_graph_cache.h` | Create | Graph cache |
| `src/compiled_graph_cache.cc` | Create | Cache implementation |
| `src/hipdnn_patterns.cc` | Modify | Add compilation logic |

---

## Phase 5: Lowering and Bufferization

**Goal**: Lower the IR to memref form for execution.

### 5.1 Lowering Pipeline

```
func.func @compiled_program {
  hipdnn.executable @graph1(...)
  hipdnn.executable @graph2(...)
}
        │
        ▼
    Convert Torch types to builtin tensor
        │
        ▼
    Destination-passing style transform
        │
        ▼
    One-shot bufferization
        │
        ▼
func.func @compiled_program {
  call @execute_hipdnn_graph("graph1", %buf0, %buf1, %out0)
  call @execute_hipdnn_graph("graph2", %out0, %buf2, %out1)
}
```

### 5.2 Custom Lowering Passes

```cpp
// include/hipdnn_ep/hipdnn_lowering.h
namespace hipdnn_ep {

// Convert torch types to standard tensor types
std::unique_ptr<mlir::Pass> createTorchToTensorPass();

// Add destination operands for hipdnn.executable ops
std::unique_ptr<mlir::Pass> createDestinationPassingPass();

// Lower hipdnn.executable to func.call @execute_hipdnn_graph
std::unique_ptr<mlir::Pass> createHipDNNExecutableToCallPass();

// Full lowering pipeline
void buildHipDNNLoweringPipeline(mlir::OpPassManager& pm);

}  // namespace hipdnn_ep
```

### 5.3 Files Changed/Added

| File | Action | Description |
|------|--------|-------------|
| `include/hipdnn_ep/hipdnn_lowering.h` | Create | Lowering passes |
| `src/hipdnn_lowering.cc` | Create | Pass implementations |

---

## Phase 6: Runtime Execution

**Goal**: Execute the compiled memref program.

### 6.1 Execution Options

**Option A: MLIR Interpreter (Recommended for Phase 1)**
- Use MLIR's execution engine for simple interpretation
- Lower overhead for small programs
- Easier debugging

**Option B: JIT Compilation (Future)**
- Use LLVM OrcJIT for native code generation
- Better performance for large programs
- More complex setup

### 6.2 Executor Interface

```cpp
// include/hipdnn_ep/executor.h
namespace hipdnn_ep {

class Executor {
public:
  Executor(mlir::ModuleOp module, CompiledGraphCache& cache);

  // Execute with given input/output buffers
  OrtStatus* Execute(OrtKernelContext* ctx);

private:
  // External function implementations
  static void ExecuteHipDNNGraph(const char* name, void** buffers);

  mlir::ModuleOp module_;
  CompiledGraphCache& cache_;
  std::unique_ptr<mlir::ExecutionEngine> engine_;
};

}  // namespace hipdnn_ep
```

### 6.3 Memory Interface

The executor needs to:
1. Map ORT tensors to memref descriptors
2. Allocate workspace memory via HIP
3. Pass buffer pointers to hipDNN execution

### 6.4 Files Changed/Added

| File | Action | Description |
|------|--------|-------------|
| `include/hipdnn_ep/executor.h` | Create | Executor header |
| `src/executor.cc` | Create | Execution implementation |

---

## Phase 7: Integration

**Goal**: Wire everything together in the existing Kernel class.

### 7.1 Modified Kernel Flow

```cpp
OrtStatus* Kernel::BuildAndCompile(Ort::ConstGraph graph) {
#ifdef HIPDNN_EP_HAS_TORCH_MLIR
  // New path: Build IR, apply patterns, lower, create executor
  IRBuilder ir_builder(mlir_context_);
  auto module = ir_builder.BuildModule(inputs, outputs, nodes);

  mlir::PassManager pm(&mlir_context_);
  buildHipDNNOffloadPipeline(pm, compiled_cache_);
  buildHipDNNLoweringPipeline(pm);
  pm.run(module);

  executor_ = std::make_unique<Executor>(module, compiled_cache_);
  return nullptr;
#else
  // Existing path: Direct hipDNN/BlasLT graph building
  // ... existing code ...
#endif
}

OrtStatus* Kernel::Execute(OrtKernelContext* ctx) {
#ifdef HIPDNN_EP_HAS_TORCH_MLIR
  return executor_->Execute(ctx);
#else
  // ... existing code ...
#endif
}
```

### 7.2 Files Changed

| File | Action | Description |
|------|--------|-------------|
| `include/hipdnn_ep/kernel.h` | Modify | Add MLIR members |
| `src/kernel.cc` | Modify | Conditional MLIR path |

---

## Timeline Estimate

| Phase | Description | Dependencies | Effort |
|-------|-------------|--------------|--------|
| 1 | Build Infrastructure | None | 1-2 weeks |
| 2 | IR Builder | Phase 1 | 2-3 weeks |
| 3 | Pattern Matching | Phase 2 | 1-2 weeks |
| 4 | hipDNN Compilation | Phase 3 | 1-2 weeks |
| 5 | Lowering & Bufferization | Phase 4 | 2-3 weeks |
| 6 | Runtime Execution | Phase 5 | 2-3 weeks |
| 7 | Integration | All phases | 1 week |

**Total**: 10-16 weeks for full implementation

---

## Testing Strategy

### Unit Tests

- IR builder: Verify correct Torch-MLIR generation from synthetic graphs
- Patterns: Test pattern matching on known IR snippets
- Lowering: Verify correct memref output

### Integration Tests

- End-to-end: Run existing test models through new pipeline
- Comparison: Verify results match existing hipDNN path
- Performance: Benchmark against baseline

### Test Files

| File | Description |
|------|-------------|
| `test/ir_builder_test.cc` | IR construction tests |
| `test/patterns_test.cc` | Pattern matching tests |
| `test/lowering_test.cc` | Lowering pass tests |
| `test/executor_test.cc` | Execution tests |
| `test/e2e_mlir_test.cc` | End-to-end integration |

---

## Milestones

### Milestone 1: "Hello MLIR"
- [ ] torch-mlir submodule added and building
- [ ] Simple IR construction test passing
- [ ] CI green for MLIR build

### Milestone 2: "Single Op"
- [ ] Conv2D compiles and executes through MLIR path
- [ ] Results match existing implementation
- [ ] Test coverage for single operations

### Milestone 3: "Multi-Op Graph"
- [ ] Two-layer MLP (from Report.md) compiles and executes
- [ ] Memory planning via bufferization working
- [ ] Performance within 10% of baseline

### Milestone 4: "Feature Parity"
- [ ] All currently supported ops work through MLIR
- [ ] Existing tests pass with MLIR enabled
- [ ] Documentation complete

---

## Open Questions

1. **hipDNN Graph API Evolution**: Should we wait for hipDNN to support more
   operations before this work, or proceed now?

2. **MLIR Version Pinning**: How do we handle torch-mlir version updates?
   Quarterly sync? Pin indefinitely?

3. **Custom Dialect**: Should we create a `hipdnn` MLIR dialect for better
   tooling, or rely on `torch.operator` approach?

4. **Shared vs. Static MLIR**: What's the deployment target? Affects linking
   strategy significantly.

---

## References

- [Torch-MLIR Repository](https://github.com/llvm/torch-mlir)
- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [MLIR Execution Engine](https://mlir.llvm.org/docs/ExecutionEngine/)
- [hipDNN Documentation](internal)
