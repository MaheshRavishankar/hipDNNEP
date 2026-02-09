//===-- HipDNNDialect.td - HipDNN dialect definition -------*- tablegen -*-===//
//
// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_DIALECT_TD
#define HIPDNN_DIALECT_TD

include "mlir/IR/OpBase.td"

def HipDNN_Dialect : Dialect {
  let name = "hipdnn";
  let cppNamespace = "::hipdnn_ep::hipdnn";
  let description = [{
    Dialect for hipDNN compiled graph execution operations.

    This dialect provides ops that represent pre-compiled hipDNN graph
    executions in the MLIR pipeline. The key op is `hipdnn.execute`, which
    implements DestinationStyleOpInterface for clean bufferization.
  }];
}

class HipDNN_Op<string mnemonic, list<Trait> traits = []>
    : Op<HipDNN_Dialect, mnemonic, traits>;

#endif // HIPDNN_DIALECT_TD
