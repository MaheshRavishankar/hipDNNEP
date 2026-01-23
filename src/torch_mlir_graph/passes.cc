// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/torch_mlir_graph/passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"

namespace hipdnn_ep {

#define GEN_PASS_REGISTRATION
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

void registerPasses() {
  registerHipDNNEpPasses();
}

bool runHipDNNOffloadPipeline(mlir::OwningOpRef<mlir::ModuleOp>& module) {
  mlir::PassManager pm(module->getContext());

  // Step 1: Convert onnx.* ops to aten.* ops
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::torch::onnx_c::createTorchOnnxToTorchPass());

  // Step 2: Apply hipDNN offload pass
  pm.addNestedPass<mlir::func::FuncOp>(createHipDNNOffloadPass());

  if (mlir::failed(pm.run(*module))) {
    return false;
  }
  return true;
}

}  // namespace hipdnn_ep
