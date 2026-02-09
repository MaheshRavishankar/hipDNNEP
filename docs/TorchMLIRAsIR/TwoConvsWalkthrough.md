# Walkthrough: Two Convolutions Through the Offload Pipeline

This document traces a two-convolution example through the
`--hipdnn-offload-pipeline` using the `hipdnn-ep-opt` tool, showing the IR
after each important transformation.

## The Example

Two sequential 2D convolutions without bias:

```
Conv0: [1,3,32,32] input × [16,3,3,3] weight → [1,16,30,30]
Conv1: [1,16,30,30] (output of Conv0) × [32,16,3,3] weight → [1,32,28,28]
```

Both use stride=1, padding=0, dilation=1, groups=1.

## Running the Pipeline

Save the input IR below to `two_convs.mlir`, then run:

```bash
hipdnn-ep-opt \
  --hipdnn-offload-pipeline \
  --mlir-print-ir-after-all \
  --mlir-print-local-scope \
  --mlir-disable-threading \
  two_convs.mlir
```

This prints the IR after every pass. The sections below show the key
stages.

## Input IR

The starting point is two `torch.aten.conv2d` ops in the Torch dialect.
Constants for stride, padding, dilation, and groups are shared between the
two convolutions.

```mlir
func.func @two_convs(
    %arg0: !torch.vtensor<[1,3,32,32],f32>,
    %arg1: !torch.vtensor<[16,3,3,3],f32>,
    %arg2: !torch.vtensor<[16],f32>,
    %arg3: !torch.vtensor<[32,16,3,3],f32>,
    %arg4: !torch.vtensor<[32],f32>)
    -> !torch.vtensor<[1,32,28,28],f32>
    attributes {torch.onnx_meta.opset_version = 14 : si64} {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %stride = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0
      : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.conv2d %arg0, %arg1, %arg2,
      %stride, %padding, %dilation, %int1
      : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>,
        !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>,
        !torch.list<int>, !torch.int
      -> !torch.vtensor<[1,16,30,30],f32>
  %1 = torch.aten.conv2d %0, %arg3, %arg4,
      %stride, %padding, %dilation, %int1
      : !torch.vtensor<[1,16,30,30],f32>, !torch.vtensor<[32,16,3,3],f32>,
        !torch.vtensor<[32],f32>, !torch.list<int>, !torch.list<int>,
        !torch.list<int>, !torch.int
      -> !torch.vtensor<[1,32,28,28],f32>
  return %1 : !torch.vtensor<[1,32,28,28],f32>
}
```

## After HipDNN Offload (Step 3)

**Pass:** `--hipdnn-offload`

Each `torch.aten.conv2d` is outlined into its own `torch.operator "hipdnn.graph"`
region. Tensor operands (input, weight, bias) become block arguments to the region.
Non-tensor values (constants, list constructs) are **cloned** into each region so
the graph is self-contained.

```mlir
func.func @two_convs(
    %arg0: !torch.vtensor<[1,3,32,32],f32>,
    %arg1: !torch.vtensor<[16,3,3,3],f32>,
    %arg2: !torch.vtensor<[16],f32>,
    %arg3: !torch.vtensor<[32,16,3,3],f32>,
    %arg4: !torch.vtensor<[32],f32>)
    -> !torch.vtensor<[1,32,28,28],f32> {

  // --- First convolution outlined into hipdnn.graph ---
  %0 = torch.operator "hipdnn.graph"(%arg0, %arg1, %arg2)
      : (!torch.vtensor<[1,3,32,32],f32>,
         !torch.vtensor<[16,3,3,3],f32>,
         !torch.vtensor<[16],f32>)
      -> !torch.vtensor<[1,16,30,30],f32> {
  ^bb0(%input0: !torch.vtensor<[1,3,32,32],f32>,
       %weight0: !torch.vtensor<[16,3,3,3],f32>,
       %bias0: !torch.vtensor<[16],f32>):
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %stride = torch.prim.ListConstruct %int1, %int1
        : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %int0, %int0
        : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %int1, %int1
        : (!torch.int, !torch.int) -> !torch.list<int>
    %conv0 = torch.aten.conv2d %input0, %weight0, %bias0,
        %stride, %padding, %dilation, %int1
        : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>,
          !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>,
          !torch.list<int>, !torch.int
        -> !torch.vtensor<[1,16,30,30],f32>
    torch.operator_terminator %conv0 : !torch.vtensor<[1,16,30,30],f32>
  }

  // --- Second convolution outlined into hipdnn.graph ---
  // Note: %0 (the output of the first graph) feeds in as a tensor argument.
  %1 = torch.operator "hipdnn.graph"(%0, %arg3, %arg4)
      : (!torch.vtensor<[1,16,30,30],f32>,
         !torch.vtensor<[32,16,3,3],f32>,
         !torch.vtensor<[32],f32>)
      -> !torch.vtensor<[1,32,28,28],f32> {
  ^bb0(%input1: !torch.vtensor<[1,16,30,30],f32>,
       %weight1: !torch.vtensor<[32,16,3,3],f32>,
       %bias1: !torch.vtensor<[32],f32>):
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %stride = torch.prim.ListConstruct %int1, %int1
        : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %int0, %int0
        : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %int1, %int1
        : (!torch.int, !torch.int) -> !torch.list<int>
    %conv1 = torch.aten.conv2d %input1, %weight1, %bias1,
        %stride, %padding, %dilation, %int1
        : !torch.vtensor<[1,16,30,30],f32>, !torch.vtensor<[32,16,3,3],f32>,
          !torch.vtensor<[32],f32>, !torch.list<int>, !torch.list<int>,
          !torch.list<int>, !torch.int
        -> !torch.vtensor<[1,32,28,28],f32>
    torch.operator_terminator %conv1 : !torch.vtensor<[1,32,28,28],f32>
  }

  return %1 : !torch.vtensor<[1,32,28,28],f32>
}
```

**Key observations:**
- Each conv2d is wrapped in its own `hipdnn.graph` region.
- Constants (`%int0`, `%int1`) and list constructs are cloned into each region
  independently, keeping the graphs self-contained.
- The dataflow between the two graphs is explicit: `%0` (output of the first
  graph) is passed as a tensor argument to the second graph.

## After Graph-to-Executable (Step 5)

**Pass:** `--hipdnn-graph-to-executable`

Each `hipdnn.graph` region is compiled by the hipDNN backend (via
`HipDNNGraph::Build()` and `HipDNNGraph::Compile()`). The region is replaced
with an opaque `hipdnn.executable` reference carrying a unique graph name.

```mlir
func.func @two_convs(
    %arg0: !torch.vtensor<[1,3,32,32],f32>,
    %arg1: !torch.vtensor<[16,3,3,3],f32>,
    %arg2: !torch.vtensor<[16],f32>,
    %arg3: !torch.vtensor<[32,16,3,3],f32>,
    %arg4: !torch.vtensor<[32],f32>)
    -> !torch.vtensor<[1,32,28,28],f32> {

  %0 = torch.operator "hipdnn.executable"(%arg0, %arg1, %arg2)
      {graph = "hipdnn_graph_0"}
      : (!torch.vtensor<[1,3,32,32],f32>,
         !torch.vtensor<[16,3,3,3],f32>,
         !torch.vtensor<[16],f32>)
      -> !torch.vtensor<[1,16,30,30],f32>

  %1 = torch.operator "hipdnn.executable"(%0, %arg3, %arg4)
      {graph = "hipdnn_graph_1"}
      : (!torch.vtensor<[1,16,30,30],f32>,
         !torch.vtensor<[32,16,3,3],f32>,
         !torch.vtensor<[32],f32>)
      -> !torch.vtensor<[1,32,28,28],f32>

  return %1 : !torch.vtensor<[1,32,28,28],f32>
}
```

**Key observations:**
- The graph regions are gone — replaced by flat `hipdnn.executable` ops.
- Each gets a module-unique name (`hipdnn_graph_0`, `hipdnn_graph_1`).
- The compiled graph objects are stored in the `CompiledGraphMap` for later
  execution by the EP runtime.
- Types are still `!torch.vtensor`.

## After Backend Legalize (Step 6)

**Pass:** `--hipdnn-backend-legalize`

This pass performs two conversions at once:
1. **Type lowering:** `!torch.vtensor<[...],f32>` → `tensor<...xf32>`
2. **Op conversion:** `torch.operator "hipdnn.executable"` → `hipdnn.execute`
   in Destination-Passing Style (DPS), with a `tensor.empty` for each output.

```mlir
func.func @two_convs(
    %arg0: tensor<1x3x32x32xf32>,
    %arg1: tensor<16x3x3x3xf32>,
    %arg2: tensor<16xf32>,
    %arg3: tensor<32x16x3x3xf32>,
    %arg4: tensor<32xf32>)
    -> tensor<1x32x28x28xf32> {

  %empty0 = tensor.empty() : tensor<1x16x30x30xf32>
  %0 = hipdnn.execute graph("hipdnn_graph_0")
      ins(%arg0, %arg1, %arg2 :
          tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>)
      outs(%empty0 : tensor<1x16x30x30xf32>)
      -> tensor<1x16x30x30xf32>

  %empty1 = tensor.empty() : tensor<1x32x28x28xf32>
  %1 = hipdnn.execute graph("hipdnn_graph_1")
      ins(%0, %arg3, %arg4 :
          tensor<1x16x30x30xf32>, tensor<32x16x3x3xf32>, tensor<32xf32>)
      outs(%empty1 : tensor<1x32x28x28xf32>)
      -> tensor<1x32x28x28xf32>

  return %1 : tensor<1x32x28x28xf32>
}
```

**Key observations:**
- All torch types are gone — the module now uses standard MLIR `tensor` types.
- `hipdnn.execute` follows DPS: it takes explicit `outs` operands and returns
  the results as tensors.
- `tensor.empty()` allocates the output destinations. These are placeholders
  that will be materialized as real allocations during bufferization.

## Final IR (After Bufferize + Finalize MemRefs, Steps 8–9)

**Passes:** `--one-shot-bufferize` then `--hipdnn-finalize-memrefs`

Bufferization converts the tensor program to a memref program. Then
`FinalizeMemRefs` promotes returned `memref.alloc` results to function
arguments — so the caller (the ORT runtime) provides the output buffer.

```mlir
func.func @two_convs(
    %arg0: memref<1x3x32x32xf32>,
    %arg1: memref<16x3x3x3xf32>,
    %arg2: memref<16xf32>,
    %arg3: memref<32x16x3x3xf32>,
    %arg4: memref<32xf32>,
    %arg5: memref<1x32x28x28xf32>) {

  // Intermediate buffer — allocated internally, not visible to caller.
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16x30x30xf32>

  hipdnn.execute graph("hipdnn_graph_0")
      ins(%arg0, %arg1, %arg2 :
          memref<1x3x32x32xf32>, memref<16x3x3x3xf32>, memref<16xf32>)
      outs(%alloc : memref<1x16x30x30xf32>)

  hipdnn.execute graph("hipdnn_graph_1")
      ins(%alloc, %arg3, %arg4 :
          memref<1x16x30x30xf32>, memref<32x16x3x3xf32>, memref<32xf32>)
      outs(%arg5 : memref<1x32x28x28xf32>)

  return
}
```

**Key observations:**
- **Intermediate buffer stays internal:** The `memref.alloc` for the
  intermediate result (Conv0 output / Conv1 input) is allocated inside the
  function. It is not returned, so `FinalizeMemRefs` leaves it alone.
- **Final output is promoted to an argument:** The second convolution's output
  was originally a `memref.alloc` that was returned. `FinalizeMemRefs` promotes
  it to `%arg5` and removes the return value from the function signature.
- **No return values:** The function now returns `void`. All outputs are written
  to caller-provided memref arguments.
- **Ready for runtime:** The EP runtime calls this function passing 5 input
  memrefs (input, two weight/bias pairs) and 1 output memref. The runtime
  manages GPU memory for all of them.

## Pipeline Summary

| Step | Pass | What changes |
|------|------|-------------|
| 1 | `TorchOnnxToTorch` | `onnx.*` → `torch.aten.*` (skipped here — input is already Torch) |
| 2 | `CSE` | Deduplicates shared constants and list constructs |
| 3 | **`HipDNNOffload`** | Each conv2d outlined into a `hipdnn.graph` region |
| 4 | `Canonicalize` + `CSE` | Cleans up dead ops; deduplicates inside regions |
| 5 | **`GraphToExecutable`** | Compiles each graph region → `hipdnn.executable` with unique name |
| 6 | **`BackendLegalize`** | `!torch.vtensor` → `tensor`, `hipdnn.executable` → `hipdnn.execute` (DPS) |
| 7 | `EmptyTensorElimination` | Folds `tensor.empty` into DPS destinations where possible |
| 8 | **`OneShotBufferize`** | `tensor` → `memref`, `tensor.empty` → `memref.alloc` |
| 9 | **`FinalizeMemRefs`** | Returned allocs promoted to function arguments |

## Type Progression

```
!torch.vtensor<[1,3,32,32],f32>     (Steps 1–5: Torch dialect)
        ↓
tensor<1x3x32x32xf32>               (Steps 6–7: builtin tensor)
        ↓
memref<1x3x32x32xf32>               (Steps 8–9: memref, ready for execution)
```
