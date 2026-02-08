// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/torch_mlir_graph/passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

namespace hipdnn_ep {

#define GEN_PASS_DEF_HIPDNNBACKENDLEGALIZEPASS
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ConvertHipDNNExecutableToCall
//===----------------------------------------------------------------------===//

/// Convert `torch.operator "hipdnn.executable"` ops to `func.call` ops.
///
/// The executable op only carries the original operands (e.g. input, weight).
/// The target graph function also expects DPS output arguments. This pattern
/// creates `tensor.empty` ops for those extra positions.
struct ConvertHipDNNExecutableToCall
    : public mlir::OpConversionPattern<mlir::torch::Torch::OperatorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::torch::Torch::OperatorOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (op.getName() != "hipdnn.executable")
      return mlir::failure();

    auto graphAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("graph");
    if (!graphAttr)
      return mlir::failure();

    // Convert result types first — these also determine the DPS output types
    // (createGraphDeclaration appends result types as extra input args).
    llvm::SmallVector<mlir::Type> resultTypes;
    for (auto type : op->getResultTypes()) {
      auto converted = getTypeConverter()->convertType(type);
      if (!converted)
        return mlir::failure();
      resultTypes.push_back(converted);
    }

    // Collect converted operands
    llvm::SmallVector<mlir::Value> callOperands(adaptor.getOperands());

    // Create tensor.empty for each DPS output arg.
    for (auto resultType : resultTypes) {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(resultType);
      auto empty = mlir::tensor::EmptyOp::create(
          rewriter, op.getLoc(), tensorType.getShape(),
          tensorType.getElementType());
      callOperands.push_back(empty);
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, graphAttr.getValue(), resultTypes, callOperands);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// HipDNNBackendLegalizePass
//===----------------------------------------------------------------------===//

struct HipDNNBackendLegalizePass
    : public impl::HipDNNBackendLegalizePassBase<HipDNNBackendLegalizePass> {
  void runOnOperation() override;
};

void HipDNNBackendLegalizePass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  auto* context = &getContext();

  // Step 1: Full conversion — convert func signatures, call ops, return ops,
  // and hipdnn.executable → func.call.
  {
    mlir::TypeConverter typeConverter;
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    typeConverter.addConversion([](mlir::Type type) { return type; });
    mlir::torch::TorchConversion::setupBackendTypeConversion(target,
                                                             typeConverter);

    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });

    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::tensor::EmptyOp>();

    // Custom pattern for hipdnn.executable → func.call
    patterns.add<ConvertHipDNNExecutableToCall>(typeConverter, context);

    // hipdnn.executable ops must be converted; other torch.operator ops are OK.
    target.addDynamicallyLegalOp<mlir::torch::Torch::OperatorOp>(
        [](mlir::torch::Torch::OperatorOp op) {
          return op.getName() != "hipdnn.executable";
        });

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
      return mlir::isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (mlir::failed(
            mlir::applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
}

}  // namespace

}  // namespace hipdnn_ep
