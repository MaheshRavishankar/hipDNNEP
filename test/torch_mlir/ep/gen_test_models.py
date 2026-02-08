#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.
#
# Generate simple ONNX models for lit testing and run hipdnn-onnx-to-torch-mlir.
#
# RUN: python3 %s %t %hipdnn-onnx-to-torch-mlir | %FileCheck %s

import onnx
from onnx import helper, TensorProto
import os
import subprocess
import sys


def gen_matmul_model(output_dir):
    """Generate a MatMul model: Y = A @ B

    After offload pipeline, the onnx.MatMul is converted to aten.matmul
    and outlined into a hipdnn.graph region. MatMul is supported by the
    hipDNN backend, so compilation succeeds and the hipdnn.graph is
    transformed to hipdnn.executable.

    Expected output:
    CHECK-LABEL: matmul
          CHECK: module {
          CHECK:   func.func @main
     CHECK-SAME:     (%[[A:.*]]: !torch.vtensor<[2,3],f32>,
     CHECK-SAME:      %[[B:.*]]: !torch.vtensor<[3,4],f32>,
     CHECK-SAME:      %[[OUT:.*]]: !torch.vtensor<[2,4],f32> {bufferization.writable = true})
     CHECK-SAME:     -> !torch.vtensor<[2,4],f32>
          CHECK:     %[[R:.*]] = torch.operator "hipdnn.executable"
     CHECK-SAME:       {graph = @hipdnn_graph_0}
          CHECK:     return %[[R]] : !torch.vtensor<[2,4],f32>
          CHECK:   }
          CHECK: }
    """
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])

    matmul_node = helper.make_node("MatMul", ["A", "B"], ["Y"], name="matmul")

    graph = helper.make_graph([matmul_node], "matmul_graph", [A, B], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 7

    path = os.path.join(output_dir, "matmul.onnx")
    onnx.save(model, path)
    return path


def gen_conv_model(output_dir):
    """Generate a Conv model

    After offload pipeline, the onnx.Conv is converted to torch.aten.convolution
    and outlined into a hipdnn.graph region. Since Conv is supported by hipDNN,
    compilation succeeds and the hipdnn.graph is transformed to hipdnn.executable.

    Expected output:
    CHECK-LABEL: conv
          CHECK: module {
          CHECK:   func.func @main
     CHECK-SAME:     (%[[X:.*]]: !torch.vtensor<[1,1,8,8],f32>,
     CHECK-SAME:      %[[W:.*]]: !torch.vtensor<[1,1,3,3],f32>,
     CHECK-SAME:      %[[OUT:.*]]: !torch.vtensor<[1,1,6,6],f32> {bufferization.writable = true})
     CHECK-SAME:     -> !torch.vtensor<[1,1,6,6],f32>
          CHECK:     %[[R:.*]] = torch.operator "hipdnn.executable"
     CHECK-SAME:       {graph = @hipdnn_graph_0}
          CHECK:     return %[[R]] : !torch.vtensor<[1,1,6,6],f32>
          CHECK:   }
          CHECK: }
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 8, 8])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 6, 6])

    conv_node = helper.make_node(
        "Conv",
        ["X", "W"],
        ["Y"],
        name="conv",
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    graph = helper.make_graph([conv_node], "conv_graph", [X, W], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 7

    path = os.path.join(output_dir, "conv.onnx")
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output_dir> <tool_path>", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[1]
    tool_path = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    # Generate and test each model
    models = [
        ("matmul", gen_matmul_model),
        ("conv", gen_conv_model),
    ]

    for name, gen_func in models:
        model_path = gen_func(output_dir)

        # Run tool and pipe to FileCheck
        print(f"// {name}")  # Label for CHECK-LABEL
        tool_result = subprocess.run(
            [tool_path, model_path],
            capture_output=True,
            text=True
        )
        if tool_result.returncode != 0:
            print(tool_result.stderr, file=sys.stderr)
            sys.exit(1)
        print(tool_result.stdout)

    # Now run FileCheck on this script with the combined output
    # FileCheck is run by the RUN line, we just output the MLIR
