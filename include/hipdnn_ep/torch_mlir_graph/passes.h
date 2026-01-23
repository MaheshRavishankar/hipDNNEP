// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

namespace mlir {
class ModuleOp;
class Pass;
template <typename T>
class OwningOpRef;
}  // namespace mlir

namespace hipdnn_ep {

#define GEN_PASS_DECL_HIPDNNOFFLOADPASS
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

/// Register all HipDNN EP passes with the MLIR pass registry.
/// This is used by the hipdnn-ep-opt tool.
void registerPasses();

/// Run the full hipDNN offload pipeline on the module:
/// 1. TorchOnnxToTorch conversion (onnx.* → aten.*)
/// 2. HipDNN offload patterns (aten.* → hipdnn.graph regions)
///
/// @param module The module to transform (modified in place)
/// @return true on success, false on failure
bool runHipDNNOffloadPipeline(mlir::OwningOpRef<mlir::ModuleOp>& module);

}  // namespace hipdnn_ep
