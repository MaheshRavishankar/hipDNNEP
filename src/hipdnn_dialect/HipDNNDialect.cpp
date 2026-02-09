// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#include "hipdnn_ep/hipdnn_dialect/HipDNNDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#include "hipdnn_ep/hipdnn_dialect/HipDNNDialect.cpp.inc"

namespace hipdnn_ep {
namespace hipdnn {

void HipDNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hipdnn_ep/hipdnn_dialect/HipDNNOps.cpp.inc"
      >();
}

}  // namespace hipdnn
}  // namespace hipdnn_ep

#define GET_OP_CLASSES
#include "hipdnn_ep/hipdnn_dialect/HipDNNOps.cpp.inc"
