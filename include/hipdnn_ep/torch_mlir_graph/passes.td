//===-- passes.td - HipDNN pass definitions ----------------*- tablegen -*-===//
//
// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_TORCH_MLIR_GRAPH_PASSES
#define HIPDNN_EP_TORCH_MLIR_GRAPH_PASSES

include "mlir/Pass/PassBase.td"

def HipDNNOffloadPass : Pass<"hipdnn-offload", "mlir::func::FuncOp"> {
  let summary = "Outline supported torch.aten ops into hipdnn.graph regions";
  let description = [{
    This pass walks the IR and outlines supported torch.aten operations
    (matmul, mm, addmm, conv2d) into torch.operator "hipdnn.graph" regions.
    These regions can then be compiled using hipDNN or hipBLAS-LT.

    Example transformation:
    ```mlir
    // Before:
    %0 = torch.aten.matmul %a, %b : ... -> ...

    // After:
    %0 = torch.operator "hipdnn.graph"(%a, %b) {
    ^bb0(%arg0, %arg1):
      %1 = torch.aten.matmul %arg0, %arg1 : ... -> ...
      torch.operator_terminator %1
    }
    ```
  }];

  let dependentDialects = [
    "mlir::torch::Torch::TorchDialect",
    "mlir::func::FuncDialect"
  ];
}

def HipDNNGraphToExecutablePass : Pass<"hipdnn-graph-to-executable", "mlir::ModuleOp"> {
  let summary = "Convert hipdnn.graph operations to hipdnn.executable operations";
  let description = [{
    This pass converts `torch.operator "hipdnn.graph"` operations (which contain
    regions with torch.aten ops) to `torch.operator "hipdnn.executable"` operations
    (which reference pre-compiled hipDNN graphs via symbol references).

    This pass compiles each hipdnn.graph region using hipDNN, stores the compiled
    graph for later execution, and creates a `func.func private` declaration at
    module scope for each compiled graph. Graph names are globally unique across
    all functions in the module.

    Example transformation:
    ```mlir
    // Before:
    func.func @main(%input: !torch.vtensor<[1,3,32,32],f32>,
                     %weight: !torch.vtensor<[16,3,3,3],f32>)
        -> !torch.vtensor<[1,16,30,30],f32> {
      %0 = torch.operator "hipdnn.graph"(%input, %weight) ({
      ^bb0(%arg0, %arg1):
        %1 = torch.aten.convolution %arg0, %arg1, ... -> ...
        torch.operator_terminator %1
      }) : ...
      return %0 : !torch.vtensor<[1,16,30,30],f32>
    }

    // After:
    func.func private @hipdnn_graph_0(
        !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>)
        -> !torch.vtensor<[1,16,30,30],f32>
    func.func @main(%input: !torch.vtensor<[1,3,32,32],f32>,
                     %weight: !torch.vtensor<[16,3,3,3],f32>)
        -> !torch.vtensor<[1,16,30,30],f32> {
      %0 = torch.operator "hipdnn.executable"(%input, %weight)
          {graph = @hipdnn_graph_0} : ...
      return %0 : !torch.vtensor<[1,16,30,30],f32>
    }
    ```
  }];

  let dependentDialects = [
    "mlir::torch::Torch::TorchDialect",
    "mlir::func::FuncDialect"
  ];

  // Use custom constructor - prevents tablegen from generating default create function
  let constructor = "::hipdnn_ep::createHipDNNGraphToExecutablePass()";
}

def HipDNNBackendLegalizePass
    : Pass<"hipdnn-backend-legalize", "mlir::ModuleOp"> {
  let summary = "Lower torch types and hipdnn.executable ops to standard MLIR";
  let description = [{
    This pass converts torch dialect types to builtin MLIR types and replaces
    `torch.operator "hipdnn.executable"` operations with `func.call` operations
    that reference the compiled graph function declarations.

    The type conversion uses torch-mlir's backend type conversion infrastructure
    to lower `!torch.vtensor` to `tensor`, `!torch.int` to `i64`, etc.

    For each `hipdnn.executable` op, the pass creates a `func.call` to the
    corresponding graph function, inserting `tensor.empty` ops for any
    destination-passing-style (DPS) output arguments that the graph function
    expects but the executable op does not provide.

    Example transformation:
    ```mlir
    // Before:
    func.func private @hipdnn_graph_0(
        !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>,
        !torch.vtensor<[1,16,30,30],f32> {bufferization.writable = true})
        -> !torch.vtensor<[1,16,30,30],f32>

    func.func @main(%arg0: !torch.vtensor<[1,3,32,32],f32>,
                     %arg1: !torch.vtensor<[16,3,3,3],f32>)
        -> !torch.vtensor<[1,16,30,30],f32> {
      %0 = torch.operator "hipdnn.executable"(%arg0, %arg1)
          {graph = @hipdnn_graph_0} : (...) -> !torch.vtensor<[1,16,30,30],f32>
      return %0 : !torch.vtensor<[1,16,30,30],f32>
    }

    // After:
    func.func private @hipdnn_graph_0(
        tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>,
        tensor<1x16x30x30xf32> {bufferization.writable = true})
        -> tensor<1x16x30x30xf32>

    func.func @main(%arg0: tensor<1x3x32x32xf32>,
                     %arg1: tensor<16x3x3x3xf32>)
        -> tensor<1x16x30x30xf32> {
      %empty = tensor.empty() : tensor<1x16x30x30xf32>
      %0 = func.call @hipdnn_graph_0(%arg0, %arg1, %empty)
          : (...) -> tensor<1x16x30x30xf32>
      return %0 : tensor<1x16x30x30xf32>
    }
    ```
  }];

  let dependentDialects = [
    "mlir::torch::Torch::TorchDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // HIPDNN_EP_TORCH_MLIR_GRAPH_PASSES
