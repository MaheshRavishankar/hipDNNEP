// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/torch_mlir_graph/passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace hipdnn_ep {

#define GEN_PASS_DEF_HIPDNNFINALIZEMEMREFSPASS
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

namespace {

struct HipDNNFinalizeMemRefsPass
    : public impl::HipDNNFinalizeMemRefsPassBase<HipDNNFinalizeMemRefsPass> {
  void runOnOperation() override;
};

void HipDNNFinalizeMemRefsPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::IRRewriter rewriter(module.getContext());

  module.walk([&](mlir::func::FuncOp funcOp) {
    auto returnOp =
        mlir::cast<mlir::func::ReturnOp>(funcOp.getBody().back().getTerminator());

    // Collect indices of return operands defined by memref.alloc.
    llvm::SmallVector<unsigned> promotedIndices;
    for (unsigned i = 0, e = returnOp.getNumOperands(); i < e; ++i) {
      if (returnOp.getOperand(i).getDefiningOp<mlir::memref::AllocOp>())
        promotedIndices.push_back(i);
    }

    if (promotedIndices.empty())
      return;

    mlir::Block& entryBlock = funcOp.getBody().front();

    // For each promoted alloc, add a new block argument and replace uses.
    for (unsigned idx : promotedIndices) {
      mlir::Value allocVal = returnOp.getOperand(idx);
      auto allocOp = allocVal.getDefiningOp<mlir::memref::AllocOp>();

      // Add a new block argument with the same memref type.
      mlir::BlockArgument newArg =
          entryBlock.addArgument(allocVal.getType(), allocOp.getLoc());

      // Replace all uses of the alloc with the new argument.
      rewriter.replaceAllUsesWith(allocVal, newArg);

      // Erase the alloc op.
      rewriter.eraseOp(allocOp);
    }

    // Build a new return without the promoted operands.
    llvm::SmallVector<mlir::Value> newReturnOperands;
    for (unsigned i = 0, e = returnOp.getNumOperands(); i < e; ++i) {
      if (!llvm::is_contained(promotedIndices, i))
        newReturnOperands.push_back(returnOp.getOperand(i));
    }
    rewriter.setInsertionPoint(returnOp);
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returnOp,
                                                      newReturnOperands);

    // Update the function type: add new argument types, remove promoted results.
    llvm::SmallVector<mlir::Type> resultTypes(
        funcOp.getFunctionType().getResults());

    llvm::SmallVector<mlir::Type> inputTypes;
    for (auto arg : entryBlock.getArguments())
      inputTypes.push_back(arg.getType());

    // Remove promoted result types (in reverse).
    for (auto it = promotedIndices.rbegin(); it != promotedIndices.rend(); ++it)
      resultTypes.erase(resultTypes.begin() + *it);

    rewriter.modifyOpInPlace(funcOp, [&] {
      funcOp.setFunctionType(
          mlir::FunctionType::get(funcOp.getContext(), inputTypes, resultTypes));
    });
  });
}

}  // namespace

}  // namespace hipdnn_ep
