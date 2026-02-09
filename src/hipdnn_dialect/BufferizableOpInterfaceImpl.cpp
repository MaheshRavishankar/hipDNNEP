// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/hipdnn_dialect/BufferizableOpInterfaceImpl.h"

#include "hipdnn_ep/hipdnn_dialect/HipDNNDialect.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

struct ExecuteOpInterface
    : public DstBufferizableOpInterfaceExternalModel<ExecuteOpInterface,
                                                      hipdnn_ep::hipdnn::ExecuteOp> {
  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const BufferizationOptions& options,
                          BufferizationState& state) const {
    auto executeOp = cast<hipdnn_ep::hipdnn::ExecuteOp>(op);

    // Get buffers for inputs
    SmallVector<Value> newInputs;
    for (auto input : executeOp.getInputs()) {
      FailureOr<Value> buffer = getBuffer(rewriter, input, options, state);
      if (failed(buffer))
        return failure();
      newInputs.push_back(*buffer);
    }

    // Get buffers for outs (DPS outputs)
    SmallVector<Value> newOuts;
    for (auto out : executeOp.getOuts()) {
      FailureOr<Value> buffer = getBuffer(rewriter, out, options, state);
      if (failed(buffer))
        return failure();
      newOuts.push_back(*buffer);
    }

    // Create new execute op with memref operands and no tensor results.
    hipdnn_ep::hipdnn::ExecuteOp::create(
        rewriter, op->getLoc(), /*resultTypes=*/TypeRange{},
        executeOp.getGraphAttr(), newInputs, newOuts);

    // Replace tensor results with out buffers.
    replaceOpWithBufferizedValues(rewriter, op, newOuts);
    return success();
  }
};

}  // namespace

namespace hipdnn_ep {
namespace hipdnn {

void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry& registry) {
  registry.addExtension(
      +[](MLIRContext* ctx, hipdnn_ep::hipdnn::HipDNNDialect* /*dialect*/) {
        hipdnn_ep::hipdnn::ExecuteOp::attachInterface<ExecuteOpInterface>(
            *ctx);
      });
}

}  // namespace hipdnn
}  // namespace hipdnn_ep
