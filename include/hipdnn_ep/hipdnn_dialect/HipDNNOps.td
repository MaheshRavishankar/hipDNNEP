//===-- HipDNNOps.td - HipDNN op definitions ---------------*- tablegen -*-===//
//
// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_OPS_TD
#define HIPDNN_OPS_TD

include "hipdnn_ep/hipdnn_dialect/HipDNNDialect.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

//===----------------------------------------------------------------------===//
// hipdnn.execute
//===----------------------------------------------------------------------===//

def HipDNN_ExecuteOp : HipDNN_Op<"execute", [
    DestinationStyleOpInterface,
    AttrSizedOperandSegments
]> {
  let summary = "Execute a pre-compiled hipDNN graph";
  let description = [{
    Executes a pre-compiled hipDNN graph identified by the `graph` string
    attribute. The op follows destination-passing style (DPS): the `outs`
    operands provide output buffers, and results are tied 1:1 to outs.

    In tensor mode (pre-bufferization):
    ```mlir
    %empty = tensor.empty() : tensor<1x16x30x30xf32>
    %0 = hipdnn.execute graph("hipdnn_graph_0")
        ins(%input, %weight : tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>)
        outs(%empty : tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
    ```

    After bufferization (no results, output written to out buffer):
    ```mlir
    %alloc = memref.alloc() : memref<1x16x30x30xf32>
    hipdnn.execute graph("hipdnn_graph_0")
        ins(%input, %weight : memref<1x3x32x32xf32>, memref<16x3x3x3xf32>)
        outs(%alloc : memref<1x16x30x30xf32>)
    ```
  }];

  let arguments = (ins
    StrAttr:$graph,
    Variadic<AnyShaped>:$inputs,
    Variadic<AnyShaped>:$outs
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutsMutable();
    }
  }];

  let assemblyFormat = [{
    `graph` `(` $graph `)`
    `ins` `(` $inputs `:` type($inputs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict (`->` type($results)^)?
  }];
}

#endif // HIPDNN_OPS_TD
