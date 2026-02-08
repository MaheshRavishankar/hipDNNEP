// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/torch_mlir_graph/passes.h"

#include <hipdnn_backend.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

namespace hipdnn_ep {

#define GEN_PASS_REGISTRATION
#include "hipdnn_ep/torch_mlir_graph/passes.h.inc"

void loadDialects(mlir::MLIRContext& ctx) {
  ctx.loadDialect<mlir::torch::Torch::TorchDialect>();
  ctx.loadDialect<mlir::torch::TorchConversion::TorchConversionDialect>();
  ctx.loadDialect<mlir::func::FuncDialect>();
  ctx.loadDialect<mlir::tensor::TensorDialect>();
}

void buildOffloadPipeline(mlir::OpPassManager& pm, hipdnnHandle_t handle,
                          CompiledGraphMap output_graphs) {
  // Step 1: Convert onnx.* ops to aten.* ops
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::torch::onnx_c::createTorchOnnxToTorchPass());

  // Step 2: Deduplicate constants and identical list constructs
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Step 3: Outline supported aten ops into hipdnn.graph regions
  pm.addNestedPass<mlir::func::FuncOp>(createHipDNNOffloadPass());

  // Step 4: Clean up dead ops left outside hipdnn.graph regions, then
  // deduplicate cloned constants inside regions.
  // Disable constant CSE to prevent hoisting constants out of regions.
  mlir::GreedyRewriteConfig canonConfig;
  canonConfig.enableConstantCSE(false);
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass(canonConfig));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Step 5: Compile hipdnn.graph regions and transform to hipdnn.executable
  pm.addPass(createHipDNNGraphToExecutablePass(handle, std::move(output_graphs)));

  // Step 6: Lower torch types and hipdnn.executable ops to builtin
  // tensor types and func.call
  pm.addPass(createHipDNNBackendLegalizePass());
}

void registerPasses() {
  registerHipDNNEpPasses();

  mlir::PassPipelineRegistration<>(
      "hipdnn-offload-pipeline",
      "Run the full hipDNN offload pipeline (onnx-to-torch, offload, "
      "graph-to-executable, backend-legalize)",
      [](mlir::OpPassManager& pm) {
        hipdnnHandle_t handle = nullptr;
        hipdnnCreate(&handle);
        buildOffloadPipeline(pm, handle, /*output_graphs=*/nullptr);
      });
}

}  // namespace hipdnn_ep
