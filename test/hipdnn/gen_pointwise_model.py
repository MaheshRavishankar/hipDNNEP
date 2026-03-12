#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate simple pointwise (Mul, Sub, Add, Div) ONNX models for testing."""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_pointwise_model(
    op_type="Mul",
    a_shape=(1, 4, 8, 8),
    b_shape=None,
    output_file="pointwise_test.onnx",
):
    """Create a pointwise binary op model: Y = op(A, B).

    When b_shape differs from a_shape, ONNX broadcasting determines Y's shape.
    """
    if b_shape is None:
        b_shape = a_shape

    # Compute broadcast output shape via numpy
    a_dummy = np.empty(a_shape)
    b_dummy = np.empty(b_shape)
    y_shape = np.broadcast_shapes(a_dummy.shape, b_dummy.shape)

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(a_shape))
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, list(b_shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(y_shape))

    node = helper.make_node(
        op_type,
        inputs=["A", "B"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_test",
        [A, B],
        [Y],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    print(f"Saved {op_type} model to {output_file}")
    print(f"  A shape: {list(a_shape)}, B shape: {list(b_shape)}, Y shape: {list(y_shape)}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["Mul", "Sub", "Add", "Div"], default="Mul")
    parser.add_argument("--output", "-o", default="pointwise_test.onnx")
    parser.add_argument(
        "--a-shape", type=int, nargs="+", default=[1, 4, 8, 8]
    )
    parser.add_argument(
        "--b-shape", type=int, nargs="+", default=None
    )
    args = parser.parse_args()

    create_pointwise_model(
        op_type=args.op,
        a_shape=tuple(args.a_shape),
        b_shape=tuple(args.b_shape) if args.b_shape else None,
        output_file=args.output,
    )
