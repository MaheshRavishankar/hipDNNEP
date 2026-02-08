// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/hipdnn_graph/hipdnn_graph.h"
#include "hipdnn_ep/torch_mlir_graph/passes.h"

#include <hipdnn_backend.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace hipdnn_ep {

#define GEN_PASS_DEF_HIPDNNGRAPHTOEXECUTABLEPASS
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

namespace {

/// Compile a single hipdnn.graph region to a HipDNNGraph.
///
/// @param region The region containing the graph body (torch.aten ops)
/// @param handle hipDNN handle for graph compilation
/// @return Compiled graph on success, nullptr on failure
static std::unique_ptr<HipDNNGraph> compileHipDNNGraph(mlir::Region& region,
                                                       hipdnnHandle_t handle) {
  if (handle == nullptr) {
    return nullptr;
  }

  auto graph = std::make_unique<HipDNNGraph>(handle);

  Status status = graph->Build(region);
  if (status.failed()) {
    return nullptr;
  }

  status = graph->Compile();
  if (status.failed()) {
    return nullptr;
  }

  return graph;
}

/// Create a private function declaration at module scope for a compiled graph.
/// The function signature is derived from the graphOp's operand/result types.
/// Returns the created declaration.
static mlir::func::FuncOp createGraphDeclaration(
    mlir::OpBuilder& moduleBuilder,
    mlir::ModuleOp module,
    mlir::torch::Torch::OperatorOp graphOp,
    const std::string& graph_name) {
  auto funcType = mlir::FunctionType::get(
      module->getContext(),
      graphOp->getOperandTypes(),
      graphOp->getResultTypes());
  auto declFunc = mlir::func::FuncOp::create(
      moduleBuilder, module->getLoc(), graph_name, funcType);
  declFunc.setVisibility(mlir::SymbolTable::Visibility::Private);
  return declFunc;
}

/// Replace a hipdnn.graph op with a hipdnn.executable op that references
/// the given symbol.
static void replaceGraphWithExecutable(
    mlir::IRRewriter& rewriter,
    mlir::torch::Torch::OperatorOp graphOp,
    mlir::FlatSymbolRefAttr graphSymbol) {
  auto loc = graphOp.getLoc();
  rewriter.setInsertionPoint(graphOp);

  auto execOp = mlir::torch::Torch::OperatorOp::create(
      rewriter, loc, graphOp->getResultTypes(),
      rewriter.getStringAttr("hipdnn.executable"), graphOp->getOperands(),
      /*numRegions=*/0);

  execOp->setAttr("graph", graphSymbol);
  rewriter.replaceOp(graphOp, execOp->getResults());
}

}  // namespace

//===----------------------------------------------------------------------===//
// HipDNN Graph to Executable Pass
//
// This pass converts hipdnn.graph operations to hipdnn.executable operations.
// It compiles each graph region using hipDNN and replaces the operation with
// an executable reference.
//
// Two modes of operation:
// 1. With explicit handle (production): Pass receives handle from caller,
//    compiled graphs are stored in the provided output map.
// 2. Self-managed handle (hipdnn-ep-opt): Pass creates its own handle.
//    If no GPU is available, compilation fails and graphs remain unchanged.
//===----------------------------------------------------------------------===//

struct HipDNNGraphToExecutablePass
    : public impl::HipDNNGraphToExecutablePassBase<HipDNNGraphToExecutablePass> {
  using Base = impl::HipDNNGraphToExecutablePassBase<HipDNNGraphToExecutablePass>;

  /// Constructor: always takes handle, owns_handle flag, and output map.
  HipDNNGraphToExecutablePass(hipdnnHandle_t handle, bool owns_handle,
                              CompiledGraphMap output_graphs)
      : Base(),
        handle_(handle),
        owns_handle_(owns_handle),
        output_graphs_(std::move(output_graphs)) {}

  ~HipDNNGraphToExecutablePass() override {
    // TODO: hipdnnDestroy crashes in current TheRock build.
    // Leaking the handle is acceptable for short-lived tools like hipdnn-ep-opt.
    // Re-enable when hipDNN fix is available.
    (void)owns_handle_;
    (void)handle_;
  }

  void runOnOperation() override;

 private:
  hipdnnHandle_t handle_;
  bool owns_handle_;
  CompiledGraphMap output_graphs_;
};

void HipDNNGraphToExecutablePass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::IRRewriter rewriter(module->getContext());
  mlir::OpBuilder moduleBuilder(module.getBody(), module.getBody()->end());
  int graph_count = 0;

  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    if (func.isDeclaration())
      continue;

    // Collect hipdnn.graph ops first (can't modify while walking)
    llvm::SmallVector<mlir::torch::Torch::OperatorOp> graphOps;
    func->walk([&](mlir::torch::Torch::OperatorOp torchOp) {
      if (torchOp.getName() == "hipdnn.graph") {
        graphOps.push_back(torchOp);
      }
    });

    // Process each graph: compile and transform to executable only on success
    for (auto graphOp : graphOps) {
      std::string graph_name = "hipdnn_graph_" + std::to_string(graph_count++);

      // Get the region containing the graph body
      if (graphOp.getNumRegions() != 1) {
        // Invalid graph structure - skip it
        continue;
      }
      mlir::Region& region = graphOp.getRegion(0);

      // Try to compile the graph (may fail if no GPU or unsupported ops)
      auto compiled_graph = compileHipDNNGraph(region, handle_);

      if (!compiled_graph) {
        // Compilation failed - leave the hipdnn.graph operation unchanged
        // This allows fallback handling at runtime
        continue;
      }

      // Compilation succeeded - store the compiled graph and transform IR
      if (output_graphs_) {
        (*output_graphs_)[graph_name] = std::move(compiled_graph);
      }

      // Create private function declaration at module scope
      auto declFunc =
          createGraphDeclaration(moduleBuilder, module, graphOp, graph_name);

      // Transform hipdnn.graph -> hipdnn.executable
      auto graphSymbol = mlir::FlatSymbolRefAttr::get(declFunc);
      replaceGraphWithExecutable(rewriter, graphOp, graphSymbol);
    }
  }
}

std::unique_ptr<mlir::Pass> createHipDNNGraphToExecutablePass() {
  hipdnnHandle_t handle = nullptr;
  hipdnnStatus_t status = hipdnnCreate(&handle);
  if (status != HIPDNN_STATUS_SUCCESS) {
    return nullptr;
  }
  return std::make_unique<HipDNNGraphToExecutablePass>(
      handle, /*owns_handle=*/true, /*output_graphs=*/nullptr);
}

std::unique_ptr<mlir::Pass> createHipDNNGraphToExecutablePass(
    hipdnnHandle_t handle, CompiledGraphMap output_graphs) {
  return std::make_unique<HipDNNGraphToExecutablePass>(
      handle, /*owns_handle=*/false, std::move(output_graphs));
}

}  // namespace hipdnn_ep
