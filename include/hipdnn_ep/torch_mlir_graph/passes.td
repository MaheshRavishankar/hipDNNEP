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

#endif // HIPDNN_EP_TORCH_MLIR_GRAPH_PASSES
