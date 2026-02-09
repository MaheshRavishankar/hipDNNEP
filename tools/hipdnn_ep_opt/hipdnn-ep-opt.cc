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

#include "hipdnn_ep/hipdnn_dialect/BufferizableOpInterfaceImpl.h"
#include "hipdnn_ep/hipdnn_dialect/HipDNNDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
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

  // Register bufferization passes
  mlir::bufferization::registerEmptyTensorEliminationPass();
  mlir::bufferization::registerOneShotBufferizePass();

  // Register dialects
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::torch::Torch::TorchDialect>();
  registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();
  registry.insert<hipdnn_ep::hipdnn::HipDNNDialect>();

  // Register bufferizable op interface external models
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  hipdnn_ep::hipdnn::registerBufferizableOpInterfaceExternalModels(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HipDNN EP MLIR optimizer driver\n", registry));
}
