// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "llvm/ADT/StringMap.h"

// Forward declarations for hipDNN
struct hipdnnHandle;
typedef hipdnnHandle* hipdnnHandle_t;

namespace mlir {
class ModuleOp;
class Pass;
template <typename T>
class OwningOpRef;
}  // namespace mlir

namespace hipdnn_ep {

class HipDNNGraph;

/// Map of compiled hipDNN graphs, keyed by graph name.
/// Used to pass compiled graphs out of the HipDNNGraphToExecutablePass.
using CompiledGraphMap = std::shared_ptr<llvm::StringMap<std::unique_ptr<HipDNNGraph>>>;

// Generate pass declarations. HipDNNGraphToExecutablePass uses a custom
// constructor (set in passes.td), so tablegen won't generate its create function.
#define GEN_PASS_DECL
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

/// Register all HipDNN EP passes with the MLIR pass registry.
/// This is used by the hipdnn-ep-opt tool.
void registerPasses();

/// Create the HipDNNGraphToExecutablePass with a self-managed hipDNN handle.
/// This is used by hipdnn-ep-opt for testing. Compiled graphs are discarded.
std::unique_ptr<mlir::Pass> createHipDNNGraphToExecutablePass();

/// Create the HipDNNGraphToExecutablePass with an explicit hipDNN handle.
/// Compiled graphs are stored in the output map for later execution.
///
/// @param handle hipDNN handle for graph compilation (must not be nullptr)
/// @param output_graphs Map to store compiled graphs (can be nullptr to discard)
std::unique_ptr<mlir::Pass> createHipDNNGraphToExecutablePass(
    hipdnnHandle_t handle, CompiledGraphMap output_graphs);

}  // namespace hipdnn_ep
