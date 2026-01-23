// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace hipdnn_ep {

#define GEN_PASS_DEF_HIPDNNOFFLOADPASS
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Outlining Helper
//===----------------------------------------------------------------------===//

/// Outline a single operation into a hipdnn.graph region.
static void outlineOpToHipDNNGraph(mlir::IRRewriter& rewriter,
                                   mlir::Operation* op) {
  auto loc = op->getLoc();
  auto resultTypes = op->getResultTypes();

  // Capture operands before moving (the range may be invalidated)
  llvm::SmallVector<mlir::Value> operandValues(op->getOperands());
  size_t numOperands = operandValues.size();

  rewriter.setInsertionPoint(op);

  // Create hipdnn.graph operator with 1 region
  auto graphOp = mlir::torch::Torch::OperatorOp::create(
      rewriter, loc, resultTypes, rewriter.getStringAttr("hipdnn.graph"),
      operandValues, /*numRegions=*/1);

  // Build the region body
  mlir::Region& region = graphOp->getRegion(0);
  mlir::Block* block = rewriter.createBlock(&region);

  // Add block arguments matching operand types
  for (mlir::Value operand : operandValues) {
    block->addArgument(operand.getType(), loc);
  }

  // Replace uses of the original op's results with graphOp's results
  // (must do this before moving, while op is still in its original location)
  rewriter.replaceAllUsesWith(op->getResults(), graphOp->getResults());

  // Move the op inside the region and remap operands to block arguments
  rewriter.moveOpBefore(op, block, block->end());
  for (size_t i = 0; i < numOperands; ++i) {
    op->setOperand(i, block->getArgument(i));
  }

  // Add terminator
  mlir::torch::Torch::OperatorTerminatorOp::create(rewriter, loc,
                                                   op->getResults());
}

//===----------------------------------------------------------------------===//
// HipDNN Offload Pass
//===----------------------------------------------------------------------===//

struct HipDNNOffloadPass
    : public impl::HipDNNOffloadPassBase<HipDNNOffloadPass> {
  void runOnOperation() override;
};

void HipDNNOffloadPass::runOnOperation() {
  mlir::func::FuncOp func = getOperation();
  mlir::IRRewriter rewriter(&getContext());

  // Collect ops to transform (we can't modify while iterating)
  llvm::SmallVector<mlir::Operation*> opsToOutline;
  func.walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::torch::Torch::AtenMmOp,
                  mlir::torch::Torch::AtenMatmulOp,
                  mlir::torch::Torch::AtenAddmmOp,
                  mlir::torch::Torch::AtenConvolutionOp,
                  mlir::torch::Torch::AtenConv2dOp>(op)) {
      opsToOutline.push_back(op);
    }
  });

  // Transform each op
  for (mlir::Operation* op : opsToOutline) {
    outlineOpToHipDNNGraph(rewriter, op);
  }
}

}  // namespace

}  // namespace hipdnn_ep
