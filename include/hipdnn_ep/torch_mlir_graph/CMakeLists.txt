# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Pass MLIR includes to TableGen without polluting global include paths
set(LLVM_TABLEGEN_FLAGS "-I${MLIR_INCLUDE_DIRS}" "-I${LLVM_INCLUDE_DIRS}")

set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls -name HipDNNEp)
add_public_tablegen_target(HipDNNEpTorchMlirPassIncGen)
