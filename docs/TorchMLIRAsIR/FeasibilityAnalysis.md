# Feasibility Analysis: Torch-MLIR as Intermediate IR for hipDNN EP

## 1. Executive Summary

This document analyzes the feasibility of using Torch-MLIR as an intermediate
representation for the hipDNN Execution Provider. The proposal offers
significant advantages for program optimization and memory planning, but comes
with substantial integration complexity and build infrastructure requirements.

**Overall Assessment**: Feasible with moderate-to-high effort. The approach is
sound from a technical perspective, but requires careful phased implementation.

## 2. Strengths of the Approach

### 2.1 Leveraging Existing Infrastructure

- **Mature Type System**: Torch-MLIR provides a well-defined type system
  (`!torch.vtensor`) that maps naturally to ONNX tensor types.
- **ONNX Import Path**: torch-mlir has existing `TorchOnnxToTorch` conversion
  passes, reducing the ONNX-to-IR translation effort.
- **Bufferization Infrastructure**: MLIR's one-shot bufferization can handle
  tensor-to-memref conversion, eliminating custom memory planning code.
- **Pattern Infrastructure**: MLIR's tablegen-based pattern matching simplifies
  writing transformation passes.

### 2.2 Optimization Opportunities

- **Full-Program Analysis**: Unlike the current per-subgraph approach, Torch-MLIR
  enables cross-subgraph optimization.
- **Memory Lifetime Analysis**: MLIR's bufferization can optimize memory
  allocation across the entire compiled program.
- **Fusion Patterns**: Generic fusion patterns can be written once and applied
  automatically.

### 2.3 Alignment with Current Implementation

The current `kernel.cc` architecture already follows a similar pattern:
- Build phase: Constructs computation graph from ONNX nodes
- Compile phase: Generates executable representation
- Execute phase: Runs the compiled graph

This maps cleanly to the proposed Torch-MLIR pipeline stages.

## 3. Technical Challenges and Risks

### 3.1 ONNX to Torch-MLIR Conversion (Medium Risk)

**Challenge**: ONNXRuntime provides ONNX graphs through its C API, not as ONNX
protobuf files. We need to:
1. Convert ORT graph representation to Torch-MLIR operations
2. Handle ONNX semantics that differ from PyTorch semantics

**Mitigation**:
- Implement a direct ORT-to-Torch-MLIR importer within the EP
- Start with the operations already supported (Conv, MatMul, Gemm)
- Leverage existing `torch-mlir::onnx_c_importer` patterns as reference

**Current Operations Supported**:
| Operation | Current EP | Torch-MLIR Equivalent |
|-----------|------------|----------------------|
| Conv2D    | Yes        | `torch.aten.conv2d`  |
| MatMul    | Yes        | `torch.aten.mm`      |
| Gemm      | Yes        | `torch.aten.addmm`   |

### 3.2 hipDNN Offload Pattern Matching (Medium Risk)

**Challenge**: Identifying sequences of operations that can be efficiently
offloaded to hipDNN requires sophisticated pattern matching.

**Current hipDNN Graph API Limitations**:
- Limited operation fusion (primarily conv + bias + activation)
- Static shape requirements
- Specific memory layout expectations (NCHW)

**Mitigation**:
- Start with 1:1 operation mapping (current behavior)
- Incrementally add fusion patterns as hipDNN support expands
- Use `torch.operator "hipdnn.graph"` as proposed for clean separation

### 3.3 Compilation Artifact Generation (High Risk)

**Challenge**: The proposal requires generating object code that can:
1. Call into the hipDNN compiled graph cache
2. Handle memory allocation via hipMalloc
3. Execute on the GPU

**Options**:

| Approach | Complexity | Runtime Overhead |
|----------|------------|------------------|
| JIT via LLVM OrcJIT | High | Low |
| Interpret memref program | Low | Medium |
| Generate C/HIP code + compile | Medium | Very Low |

**Recommendation**: Start with an interpreter-based approach for the `memref`
program. This avoids JIT complexity while proving the IR design. JIT compilation
can be added later for performance.

### 3.4 Memory Planning Integration (Medium Risk)

**Challenge**: Coordinating between:
- MLIR bufferization (intermediate buffers)
- hipDNN workspace requirements
- ONNXRuntime memory allocator

**Mitigation**:
- Use custom allocation function that routes to HIP allocator
- Query hipDNN workspace requirements during compilation
- Pre-allocate workspace as part of kernel initialization

### 3.5 Custom Operations Fallback (Medium Risk)

**Challenge**: Operations not supported by hipDNN need a fallback path.

**Options**:
1. **Fusilli/IREE plugin**: As mentioned in the proposal, use for arbitrary
   Torch-MLIR compilation
2. **CPU fallback**: Route to ONNX Runtime CPU EP
3. **rocBLAS/MIOpen direct**: Use other ROCm libraries for specific ops

**Recommendation**: Initially reject subgraphs with unsupported operations,
allowing ONNX Runtime to use other EPs. Add Fusilli integration later.

## 4. Dependency Analysis

### 4.1 torch-mlir

**Repository**: https://github.com/llvm/torch-mlir

**Integration Options**:

| Option | Pros | Cons |
|--------|------|------|
| Git submodule | Version control, reproducible | Build time, repo size |
| Find installed | Smaller repo | Deployment complexity |
| Fetch at build time | Flexible version | Network dependency |

**Recommendation**: Git submodule for development, with option to use installed
version for deployment.

**Relevant Components**:
- `torch-mlir::TorchDialect` - Core Torch dialect
- `torch-mlir::TorchConversionDialect` - Conversion utilities
- `lib/Conversion/TorchOnnxToTorch` - ONNX import patterns

### 4.2 LLVM/MLIR

**Version Requirement**: torch-mlir tracks LLVM main; we need the matching
version.

**Build Options**:
1. **Bundled with torch-mlir**: Simplest, but ~2GB+ build artifact
2. **External LLVM**: Smaller repo, complex setup
3. **Shared system LLVM**: Risk of version mismatch

**Recommendation**: Build LLVM/MLIR as part of torch-mlir submodule (option 1)
for initial development. This is the most reliable approach.

### 4.3 Dependency Tree

```
hipDNNEP
├── torch-mlir (submodule)
│   └── externals/
│       └── llvm-project (fetched or submodule)
├── ONNXRuntime headers (existing)
└── TheRock/ROCm (existing)
    ├── hipDNN
    └── hipBLAS-LT
```

## 5. Build System Impact

### 5.1 CMake Changes Required

```cmake
# New dependencies
add_subdirectory(third_party/torch-mlir EXCLUDE_FROM_ALL)

# Link against torch-mlir libraries
target_link_libraries(hipdnn_ep PRIVATE
  TorchMLIRTorchDialect
  TorchMLIRTorchToLinalg
  MLIRBufferization
  # ... additional MLIR libs
)
```

### 5.2 Build Time Impact

| Component | Approx. Build Time |
|-----------|-------------------|
| Current hipDNNEP | ~1 minute |
| LLVM/MLIR (RelWithDebInfo) | ~30-60 minutes |
| torch-mlir | ~5-10 minutes |

**Total**: Initial build increases significantly. Incremental builds remain fast
once LLVM is built.

### 5.3 Binary Size Impact

| Component | Approx. Size |
|-----------|--------------|
| Current libhipdnn_ep.so | ~500KB |
| With MLIR libs (static) | ~50-100MB |
| With MLIR libs (shared) | ~5MB + shared libs |

**Recommendation**: Use shared MLIR libraries for development; evaluate static
linking for deployment.

## 6. Runtime Considerations

### 6.1 Compilation Latency

MLIR-based compilation will add latency to the model loading phase:
- Parse/create IR: ~1-10ms per operation
- Pattern matching: ~10-100ms per subgraph
- Bufferization: ~10-50ms per function
- Code generation (if JIT): ~100-500ms

**Mitigation**:
- Cache compiled artifacts keyed by subgraph hash
- Support AOT compilation for production
- Lazy compilation (compile on first execution)

### 6.2 Memory Overhead

Additional runtime memory for:
- MLIR context and modules: ~10-50MB
- Compiled artifact cache: Depends on model size
- JIT code cache (if applicable): ~1-10MB per kernel

## 7. Roadblocks and Blockers

### 7.1 Hard Blockers

1. **torch-mlir build system complexity**: The project has historically had
   build issues on certain platforms. Need to verify ROCm/HIP compatibility.

2. **LLVM version compatibility**: torch-mlir tracks LLVM main. If ROCm/TheRock
   depends on a specific LLVM version, conflicts may arise.

### 7.2 Soft Blockers (Workarounds Exist)

1. **hipDNN operation coverage**: Limited to operations hipDNN supports.
   Workaround: reject unsupported subgraphs.

2. **Dynamic shapes**: MLIR bufferization requires static shapes by default.
   Workaround: require static shapes initially; add dynamic support later.

3. **Debugging complexity**: MLIR adds abstraction layers.
   Workaround: extensive logging, MLIR debugging tools.

## 8. Risk Summary

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Build system complexity | High | Medium | Phased integration, CI early |
| LLVM version conflicts | Medium | High | Pin versions, test matrix |
| hipDNN coverage gaps | High | Low | Graceful fallback |
| Performance regression | Medium | Medium | Benchmark suite, profiling |
| Maintenance burden | Medium | Medium | Track torch-mlir releases |

## 9. Recommendations

### 9.1 Proceed with Implementation

The approach is technically sound and aligns with industry trends. The benefits
of full-program optimization and standardized IR outweigh the integration costs.

### 9.2 Phased Approach

1. **Phase 1**: Add submodules, build infrastructure, basic IR construction
2. **Phase 2**: Implement ORT-to-Torch-MLIR conversion for supported ops
3. **Phase 3**: Implement hipDNN pattern matching and offload
4. **Phase 4**: Add bufferization and memory planning
5. **Phase 5**: Implement execution (interpreter or JIT)
6. **Phase 6**: Add custom operation fallback (Fusilli)

### 9.3 Success Criteria

- Conv2D/MatMul/Gemm operations compile and execute correctly
- Memory planning reduces peak memory usage vs. current approach
- No regression in execution performance
- Build time under 10 minutes (with pre-built LLVM)

## 10. Conclusion

Using Torch-MLIR as an intermediate IR is a feasible approach that provides
a solid foundation for future optimization work. The main challenges are build
system integration and LLVM dependency management, both of which are
surmountable with careful planning.

The recommended path forward is to proceed with a phased implementation,
starting with build infrastructure and basic IR construction before tackling
the more complex compilation and execution stages.
