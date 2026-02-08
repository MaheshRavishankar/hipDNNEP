// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
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

/// Return true for ops whose results are all torch scalar, list, or none
/// types. These ops (and their backward slice) will be cloned into the region
/// rather than captured as block arguments.
static bool shouldCloneIntoRegion(mlir::Operation* op) {
  return llvm::all_of(op->getResultTypes(), [](mlir::Type type) {
    return mlir::isa<mlir::torch::Torch::IntType, mlir::torch::Torch::BoolType,
                     mlir::torch::Torch::NoneType,
                     mlir::torch::Torch::ListType>(type);
  });
}

/// Outline a single operation into a hipdnn.graph region.
///
/// The region captures only tensor operands as block arguments.
/// Non-tensor values (constants, lists) are cloned into the region via
/// makeRegionIsolatedFromAbove.
static void outlineOpToHipDNNGraph(mlir::IRRewriter& rewriter,
                                   mlir::Operation* op) {
  auto loc = op->getLoc();
  auto resultTypes = op->getResultTypes();

  rewriter.setInsertionPoint(op);

  // Create hipdnn.graph with an empty operand list and 1 region
  auto graphOp = mlir::torch::Torch::OperatorOp::create(
      rewriter, loc, resultTypes, rewriter.getStringAttr("hipdnn.graph"),
      /*operands=*/{}, /*numRegions=*/1);

  // Build the region body
  mlir::Region& region = graphOp->getRegion(0);
  mlir::Block* block = rewriter.createBlock(&region);

  // Replace uses of the original op's results with graphOp's results
  // (must do this before moving, while op is still in its original location)
  rewriter.replaceAllUsesWith(op->getResults(), graphOp->getResults());

  // Move the op inside the region
  rewriter.moveOpBefore(op, block, block->end());

  // Add terminator
  mlir::torch::Torch::OperatorTerminatorOp::create(rewriter, loc,
                                                   op->getResults());

  // Make the region isolated from above:
  // - Ops producing torch scalar/list/none types get cloned into region
  // - Remaining values (tensors) become block args on the region
  auto captured = mlir::makeRegionIsolatedFromAbove(rewriter, region,
                                                    shouldCloneIntoRegion);

  // Add captured values as operands to the hipdnn.graph op
  rewriter.modifyOpInPlace(graphOp, [&]() {
    graphOp.getOperandsMutable().append(captured);
  });
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
