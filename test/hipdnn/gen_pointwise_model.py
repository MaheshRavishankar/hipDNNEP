#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate simple pointwise ONNX models (Mul, Sub, Add, Div) for testing."""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_pointwise_model(op_type, shape, output_file):
    """Create a binary pointwise model: Y = op(A, B)."""

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, shape)
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)

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
    print(f"Saved {op_type} model to {output_file}  shape={shape}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-d", default=".")
    args = parser.parse_args()

    shape = [2, 3, 4, 4]

    for op in ["Mul", "Sub", "Add", "Div"]:
        path = os.path.join(args.output_dir, f"{op.lower()}_test.onnx")
        create_pointwise_model(op, shape, path)

    print("Done!")
