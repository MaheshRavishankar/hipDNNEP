#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate simple unary pointwise (Sigmoid) ONNX models for testing."""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_unary_pointwise_model(
    op_type="Sigmoid",
    x_shape=(1, 4, 8, 8),
    output_file="sigmoid_test.onnx",
):
    """Create a unary pointwise op model: Y = op(X)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(x_shape))

    node = helper.make_node(
        op_type,
        inputs=["X"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_test",
        [X],
        [Y],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    print(f"Saved {op_type} model to {output_file}")
    print(f"  X shape: {list(x_shape)}, Y shape: {list(x_shape)}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["Sigmoid"], default="Sigmoid")
    parser.add_argument("--output", "-o", default="sigmoid_test.onnx")
    parser.add_argument(
        "--x-shape", type=int, nargs="+", default=[1, 4, 8, 8]
    )
    args = parser.parse_args()

    create_unary_pointwise_model(
        op_type=args.op,
        x_shape=tuple(args.x_shape),
        output_file=args.output,
    )
