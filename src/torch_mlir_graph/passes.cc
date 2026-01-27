// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/torch_mlir_graph/passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace hipdnn_ep {

#define GEN_PASS_REGISTRATION
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

void registerPasses() {
  registerHipDNNEpPasses();
}

}  // namespace hipdnn_ep
