//===- HipDNNDialect.h - HipDNN dialect declaration -------------*- C++ -*-===//
//
// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"  // Required by property-based ops
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "hipdnn_ep/hipdnn_dialect/HipDNNDialect.h.inc"

#define GET_OP_CLASSES
#include "hipdnn_ep/hipdnn_dialect/HipDNNOps.h.inc"
