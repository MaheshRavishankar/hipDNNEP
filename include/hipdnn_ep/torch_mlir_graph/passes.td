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
    (which reference pre-compiled hipDNN graphs by name via a string attribute).

    This pass compiles each hipdnn.graph region using hipDNN, stores the compiled
    graph for later execution, and replaces the graph op with an executable op
    carrying a `graph = "name"` string attribute. Graph names are globally unique
    across all functions in the module.

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
    func.func @main(%input: !torch.vtensor<[1,3,32,32],f32>,
                     %weight: !torch.vtensor<[16,3,3,3],f32>)
        -> !torch.vtensor<[1,16,30,30],f32> {
      %0 = torch.operator "hipdnn.executable"(%input, %weight)
          {graph = "hipdnn_graph_0"} : ...
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
    `torch.operator "hipdnn.executable"` operations with `hipdnn.execute`
    operations that follow destination-passing style (DPS).

    The type conversion uses torch-mlir's backend type conversion infrastructure
    to lower `!torch.vtensor` to `tensor`, `!torch.int` to `i64`, etc.

    For each `hipdnn.executable` op, the pass creates an `hipdnn.execute` op
    with `tensor.empty` ops for the DPS init (output) operands.

    Example transformation:
    ```mlir
    // Before:
    func.func @main(%arg0: !torch.vtensor<[1,3,32,32],f32>,
                     %arg1: !torch.vtensor<[16,3,3,3],f32>)
        -> !torch.vtensor<[1,16,30,30],f32> {
      %0 = torch.operator "hipdnn.executable"(%arg0, %arg1)
          {graph = "hipdnn_graph_0"} : (...) -> !torch.vtensor<[1,16,30,30],f32>
      return %0 : !torch.vtensor<[1,16,30,30],f32>
    }

    // After:
    func.func @main(%arg0: tensor<1x3x32x32xf32>,
                     %arg1: tensor<16x3x3x3xf32>)
        -> tensor<1x16x30x30xf32> {
      %empty = tensor.empty() : tensor<1x16x30x30xf32>
      %0 = hipdnn.execute(%arg0, %arg1) inits(%empty) {graph = "hipdnn_graph_0"}
          : (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>)
            inits(tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
      return %0 : tensor<1x16x30x30xf32>
    }
    ```
  }];

  let dependentDialects = [
    "hipdnn_ep::hipdnn::HipDNNDialect",
    "mlir::torch::Torch::TorchDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // HIPDNN_EP_TORCH_MLIR_GRAPH_PASSES
