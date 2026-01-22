// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "hipdnn_ep/core/ep_utils.h"

#include <memory>

namespace hipdnn_ep {

struct IRBuilderImpl;

/// IRBuilder converts ORT subgraphs to Torch-MLIR modules.
///
/// This class takes ONNX Runtime graph information (inputs, outputs, nodes)
/// and builds an equivalent Torch-MLIR module using the Torch dialect.
/// The generated IR uses value-semantic tensor types (!torch.vtensor).
///
/// Example generated IR:
///   func.func @main(%arg0: !torch.vtensor<[1,3,224,224],f32>,
///                   %arg1: !torch.vtensor<[64,3,7,7],f32>)
///       -> !torch.vtensor<[1,64,112,112],f32> {
///     %0 = torch.aten.conv2d %arg0, %arg1, ...
///     return %0 : !torch.vtensor<[1,64,112,112],f32>
///   }
class IRBuilder {
 public:
  IRBuilder();
  ~IRBuilder();

  // Non-copyable
  IRBuilder(const IRBuilder&) = delete;
  IRBuilder& operator=(const IRBuilder&) = delete;

  /// Build a Torch-MLIR module from ORT graph components.
  ///
  /// @param inputs Graph input value infos
  /// @param outputs Graph output value infos
  /// @param nodes Graph nodes in topological order
  /// @return true on success, false on error
  bool BuildModule(const std::vector<Ort::ConstValueInfo>& inputs,
                   const std::vector<Ort::ConstValueInfo>& outputs,
                   const std::vector<Ort::ConstNode>& nodes);

  /// Print the built module to a string.
  /// Returns empty string if no module has been built.
  std::string PrintModule() const;

 private:
  std::unique_ptr<IRBuilderImpl> impl_;
};

}  // namespace hipdnn_ep
