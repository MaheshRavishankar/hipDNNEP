# Plan: Refactor to Generic Kernel Architecture

## Goal

Refactor from Conv-specific kernel handling to a generic `Kernel` class that can handle any supported operation type by iterating over graph nodes and building a hipDNN graph.

## Current Architecture

```text
CompileImpl (ep.cc)
  └─> Create ConvKernel with Conv attributes
  └─> Store in kernels_ map

CreateStateImpl (conv_kernel.cc)
  └─> Look up kernel by node name
  └─> Return as compute_state

ComputeImpl (conv_kernel.cc)
  └─> kernel->Compute()
      └─> Lazily build hipDNN graph on first call
      └─> Execute
```

## Proposed Architecture

```text
CompileImpl (ep.cc)
  └─> Iterate over nodes in graph
  └─> Extract OpInfo for each node (ConvOpInfo for Conv, etc.)
  └─> Create Kernel with list of OpInfo
  └─> Build and compile hipDNN graph
  └─> Store compiled Kernel in kernels_ map
  └─> Return generic NodeComputeInfo

CreateStateImpl (node_compute_info.cc)
  └─> Look up compiled Kernel by node name
  └─> Return Kernel as compute_state

ComputeImpl (node_compute_info.cc)
  └─> kernel->Execute()
```

## Key Design Change

Graph building and compilation now happens in **CompileImpl** (not CreateStateImpl).
This assumes hipDNN can handle dynamic shapes at compile time.

## Files to Modify/Create

### 1. New: `include/hipdnn_ep/kernel.h`

```cpp
// Base class for operation info - op-agnostic
struct OpInfo {
  std::string op_type;
  std::string node_name;

  virtual ~OpInfo() = default;

  // Factory method to create appropriate derived type
  static std::unique_ptr<OpInfo> Create(Ort::ConstNode node);
};

// Conv-specific operation info
struct ConvOpInfo : OpInfo {
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  int64_t group{1};

  // Input/output shapes for graph building
  std::vector<int64_t> x_shape;
  std::vector<int64_t> w_shape;
  ONNXTensorElementDataType dtype;
};

// Future: MatMulOpInfo, ReluOpInfo, etc.

struct Kernel {
  Kernel(const OrtApi& api, const OrtLogger& logger, hipdnnHandle_t handle);

  // Build and compile hipDNN graph from list of ops
  OrtStatus* BuildAndCompile(const std::vector<std::unique_ptr<OpInfo>>& ops);

  // Execute the compiled graph
  OrtStatus* Execute(OrtKernelContext* ctx);

private:
  OrtStatus* AddOp(const OpInfo& info);      // Dispatch to appropriate Add*Op
  OrtStatus* AddConvOp(const ConvOpInfo& info);  // Add Conv to hipDNN graph
  // Future: AddMatMulOp, AddReluOp, etc.

  const OrtApi& ort_api_;
  const OrtLogger& logger_;
  hipdnnHandle_t handle_;

  std::shared_ptr<hipdnn_frontend::graph::Graph> graph_;
  std::vector<char> workspace_;
};
```

### 2. New: `src/kernel.cc`

- `OpInfo::Create()`: Factory that creates ConvOpInfo (or other derived types) based on op_type
- `BuildAndCompile()`: iterate over ops, call AddOp for each, then compile graph
- `AddOp()`: dispatch to AddConvOp based on dynamic type (or op_type string)
- `AddConvOp()`: add conv_fprop to hipDNN graph using ConvOpInfo attributes
- Compile: validate, build_operation_graph, create_execution_plans, build_plans
- `Execute()`: get tensors, create variant_pack, call graph_->execute()

### 3. New: `include/hipdnn_ep/node_compute_info.h`

```cpp
struct NodeComputeInfo : OrtNodeComputeInfo {
  NodeComputeInfo(class HipDNNEp& ep);

  static OrtStatus* CreateStateImpl(...);   // Looks up compiled Kernel
  static OrtStatus* ComputeImpl(...);       // Calls kernel->Execute()
  static void ReleaseStateImpl(...);        // No-op (EP owns kernel)

  HipDNNEp& ep;
};
```

### 4. New: `src/node_compute_info.cc`

- CreateStateImpl: lookup compiled Kernel by node name, return as state
- ComputeImpl: cast state to Kernel*, call Execute()
- ReleaseStateImpl: no-op (Kernel is owned by EP, not per-inference)

### 5. Modify: `include/hipdnn_ep/ep.h`

- Keep `std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels_`
- Keep `Kernels()` accessor
- Forward declare `Kernel` instead of `ConvKernel`

### 6. Modify: `src/ep.cc`

- Keep IsSupportedOp/IsSupportedConv as-is
- CompileImpl:
  - Iterate over nodes in graph
  - For each node, call `OpInfo::Create(node)` to get appropriate OpInfo
  - Collect all OpInfo into a vector
  - Create Kernel, call `BuildAndCompile(ops)`
  - Store compiled Kernel in `kernels_` map
- ReleaseNodeComputeInfosImpl: update to use generic NodeComputeInfo
- Remove #include "kernels/conv_kernel.h", add #include "kernel.h" and "node_compute_info.h"

### 7. Delete: `include/hipdnn_ep/kernels/conv_kernel.h`
### 8. Delete: `src/kernels/conv_kernel.cc`

### 9. Modify: `CMakeLists.txt`
- Remove `src/kernels/conv_kernel.cc`
- Add `src/kernel.cc` and `src/node_compute_info.cc`

## Implementation Steps

### Step 1: Create OpInfo hierarchy and Kernel class header

Create `include/hipdnn_ep/kernel.h`:

- Define base `OpInfo` struct with op_type, node_name, virtual destructor
- Define `ConvOpInfo` derived struct with Conv-specific attributes and shapes
- Define `Kernel` class with `BuildAndCompile()` and `Execute()` methods
- Add `OpInfo::Create()` factory method declaration

### Step 2: Implement Kernel class

Create `src/kernel.cc`:

- `OpInfo::Create()`: Factory that reads node op_type and creates appropriate derived type
- `BuildAndCompile()`: iterate over ops, call AddOp for each, then compile
- `AddOp()`: dispatch to AddConvOp based on op_type
- `AddConvOp()`: port hipDNN conv_fprop setup from conv_kernel.cc
- Compile section: validate, build_operation_graph, create_execution_plans, build_plans
- `Execute()`: port tensor handling and graph_->execute() from conv_kernel.cc

### Step 3: Create NodeComputeInfo header

Create `include/hipdnn_ep/node_compute_info.h`:

- Inherits OrtNodeComputeInfo
- Holds reference to HipDNNEp
- Static CreateStateImpl, ComputeImpl, ReleaseStateImpl

### Step 4: Implement NodeComputeInfo

Create `src/node_compute_info.cc`:

- CreateStateImpl: lookup compiled Kernel by node name, return as state
- ComputeImpl: cast state to Kernel*, call Execute()
- ReleaseStateImpl: no-op (EP owns the Kernel)

### Step 5: Update ep.h

- Change forward declaration from `ConvKernel` to `Kernel`
- Keep `kernels_` map with type `std::unique_ptr<Kernel>`
- Keep `Kernels()` accessor

### Step 6: Update ep.cc

- Remove #include "kernels/conv_kernel.h"
- Add #include "kernel.h" and "node_compute_info.h"
- CompileImpl:
  - Iterate graph.GetNodes()
  - For each node: call `OpInfo::Create(node)` to extract attributes
  - Create Kernel, call `BuildAndCompile(ops)`
  - Store in `kernels_` map
- ReleaseNodeComputeInfosImpl: cast to NodeComputeInfo

### Step 7: Update CMakeLists.txt

- Remove: `src/kernels/conv_kernel.cc`
- Add: `src/kernel.cc`, `src/node_compute_info.cc`

### Step 8: Delete old files

- Remove `include/hipdnn_ep/kernels/conv_kernel.h`
- Remove `src/kernels/conv_kernel.cc`
- Remove empty `kernels/` directories if applicable

### Step 9: Build and Test

```bash
cmake --preset RelWithDebInfo
cmake --build --preset RelWithDebInfo
ctest --preset RelWithDebInfo
```

## Key Design Decisions

1. **OpInfo class hierarchy**: Base `OpInfo` is op-agnostic; derived classes (e.g., `ConvOpInfo`) hold op-specific attributes
2. **Graph built in CompileImpl**: hipDNN graph is built and compiled during session creation (assumes hipDNN handles dynamic shapes)
3. **Kernel stored after compilation**: Compiled Kernel is stored in EP's `kernels_` map
4. **CreateStateImpl just looks up**: No building, just returns pointer to pre-compiled Kernel
5. **Factory pattern for OpInfo**: `OpInfo::Create(node)` creates the appropriate derived type

## Benefits of This Architecture

1. **Extensibility**: Adding new ops only requires:
   - New derived OpInfo class (e.g., `MatMulOpInfo`)
   - Update `OpInfo::Create()` factory
   - New `AddXxxOp()` method in Kernel

2. **Fusion-ready**: Multiple OpInfo can be passed to `BuildAndCompile()` for Conv+Bias+Relu patterns

3. **Separation of concerns**:
   - `ep.cc`: Op support checking, OpInfo creation via factory
   - `kernel.cc`: hipDNN graph construction and execution
   - `node_compute_info.cc`: ORT callback interface (thin layer)

4. **Cleaner EP**: ep.cc no longer has Conv-specific code; uses generic `OpInfo::Create()`

5. **Efficient execution**: Graph is pre-compiled; CreateStateImpl and ComputeImpl are lightweight

## Critical Files Summary

| File                                       | Action | Description                          |
|--------------------------------------------|--------|--------------------------------------|
| `include/hipdnn_ep/kernel.h`               | Create | OpInfo hierarchy + Kernel class      |
| `src/kernel.cc`                            | Create | BuildAndCompile/Execute impl         |
| `include/hipdnn_ep/node_compute_info.h`    | Create | ORT callback interface               |
| `src/node_compute_info.cc`                 | Create | CreateState/Compute/ReleaseState     |
| `include/hipdnn_ep/ep.h`                   | Modify | Forward decl Kernel, keep kernels_   |
| `src/ep.cc`                                | Modify | Use OpInfo::Create, build in Compile |
| `CMakeLists.txt`                           | Modify | Update source list                   |
| `include/hipdnn_ep/kernels/conv_kernel.h`  | Delete | Replaced by kernel.h                 |
| `src/kernels/conv_kernel.cc`               | Delete | Replaced by kernel.cc                |
