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
    shape=(1, 4, 8, 8),
    output_file="pointwise_test.onnx",
):
    """Create a pointwise binary op model: Y = op(A, B)."""

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(shape))
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, list(shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(shape))

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
    print(f"  Shape: {list(shape)}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["Mul", "Sub", "Add", "Div"], default="Mul")
    parser.add_argument("--output", "-o", default="pointwise_test.onnx")
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[1, 4, 8, 8]
    )
    args = parser.parse_args()

    create_pointwise_model(
        op_type=args.op,
        shape=tuple(args.shape),
        output_file=args.output,
    )
