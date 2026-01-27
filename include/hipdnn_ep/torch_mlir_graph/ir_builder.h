// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "hipdnn_ep/hipdnn_graph/hipdnn_graph.h"

#include <memory>
#include <string>

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

  /// Run the hipDNN offload and compilation pipeline on the built module.
  ///
  /// This method:
  /// 1. Converts onnx.* ops to aten.* ops (TorchOnnxToTorch)
  /// 2. Outlines supported ops into hipdnn.graph regions (HipDNNOffloadPass)
  /// 3. Compiles each graph region using hipDNN
  /// 4. Stores compiled graphs in the internal cache (retrievable via GetCompiledGraph)
  /// 5. Transforms the IR: replaces hipdnn.graph ops with hipdnn.executable ops
  ///
  /// @param handle hipDNN handle for graph compilation
  /// @return true on success, false on error
  bool RunOffloadPipeline(hipdnnHandle_t handle);

  /// Get a compiled graph by name from the cache.
  /// @param name The unique name of the compiled graph
  /// @return Pointer to the compiled graph, or nullptr if not found
  HipDNNGraph* GetCompiledGraph(const std::string& name);

  /// Get the number of compiled graphs in the cache.
  size_t GetCompiledGraphCount() const;

 private:
  std::unique_ptr<IRBuilderImpl> impl_;
};

}  // namespace hipdnn_ep
