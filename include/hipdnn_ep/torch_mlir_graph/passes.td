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

def HipDNNGraphToExecutablePass : Pass<"hipdnn-graph-to-executable", "mlir::func::FuncOp"> {
  let summary = "Convert hipdnn.graph operations to hipdnn.executable operations";
  let description = [{
    This pass converts `torch.operator "hipdnn.graph"` operations (which contain
    regions with torch.aten ops) to `torch.operator "hipdnn.executable"` operations
    (which reference pre-compiled hipDNN graphs by name).

    This pass compiles each hipdnn.graph region using hipDNN and stores the
    compiled graph for later execution.

    Example transformation:
    ```mlir
    // Before:
    %0 = torch.operator "hipdnn.graph"(%input, %weight) ({
    ^bb0(%arg0, %arg1):
      %1 = torch.aten.convolution %arg0, %arg1, ... -> ...
      torch.operator_terminator %1
    }) : ...

    // After:
    %0 = torch.operator "hipdnn.executable"(%input, %weight) {graph_name = "hipdnn_graph_0"} : ...
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
