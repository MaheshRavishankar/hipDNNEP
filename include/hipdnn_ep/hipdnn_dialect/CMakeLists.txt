# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

set(LLVM_TABLEGEN_FLAGS
  "-I${MLIR_INCLUDE_DIRS}"
  "-I${LLVM_INCLUDE_DIRS}"
  "-I${CMAKE_SOURCE_DIR}/include"
)

set(LLVM_TARGET_DEFINITIONS HipDNNOps.td)
mlir_tablegen(HipDNNOps.h.inc -gen-op-decls)
mlir_tablegen(HipDNNOps.cpp.inc -gen-op-defs)
mlir_tablegen(HipDNNDialect.h.inc -gen-dialect-decls -dialect=hipdnn)
mlir_tablegen(HipDNNDialect.cpp.inc -gen-dialect-defs -dialect=hipdnn)
add_public_tablegen_target(HipDNNDialectIncGen)
