// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace mlir {
class DialectRegistry;
}

namespace hipdnn_ep {
namespace hipdnn {

void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry& registry);

}  // namespace hipdnn
}  // namespace hipdnn_ep
