//===- hipdnn-ep-opt.cc - HipDNN EP MLIR Optimizer Driver -----------------===//
//
// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Main entry point for the hipdnn-ep-opt tool. This tool can run HipDNN EP
// passes on MLIR files for testing purposes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

#include "hipdnn_ep/torch_mlir_graph/passes.h"

int main(int argc, char** argv) {
  // Register HipDNN EP passes
  hipdnn_ep::registerPasses();

  // Register core MLIR passes
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  // Register dialects
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::torch::Torch::TorchDialect>();
  registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HipDNN EP MLIR optimizer driver\n", registry));
}
