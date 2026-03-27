#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate SimplifiedLayerNormalization (RMS Norm) ONNX models for testing."""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_simplified_layer_norm_model(
    x_shape=(1, 4, 8, 8),
    axis=-1,
    epsilon=1e-5,
    output_file="simplified_layer_norm_test.onnx",
):
    """Create a SimplifiedLayerNormalization model: Y = X * Scale / sqrt(mean(X^2) + eps)."""
    # Scale shape: dimensions from axis to end of X
    rank = len(x_shape)
    if axis < 0:
        axis = rank + axis
    scale_shape = list(x_shape[axis:])

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape))
    Scale = helper.make_tensor_value_info("Scale", TensorProto.FLOAT, scale_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(x_shape))

    node = helper.make_node(
        "SimplifiedLayerNormalization",
        inputs=["X", "Scale"],
        outputs=["Y"],
        axis=axis,
        epsilon=epsilon,
    )

    graph = helper.make_graph(
        [node],
        "simplified_layer_norm_test",
        [X, Scale],
        [Y],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 1),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 8

    onnx.save(model, output_file)
    print(f"Saved SimplifiedLayerNormalization model to {output_file}")
    print(f"  X shape: {list(x_shape)}, Scale shape: {scale_shape}, Y shape: {list(x_shape)}")
    print(f"  axis: {axis}, epsilon: {epsilon}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="simplified_layer_norm_test.onnx")
    parser.add_argument(
        "--x-shape", type=int, nargs="+", default=[1, 4, 8, 8]
    )
    parser.add_argument("--axis", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()

    create_simplified_layer_norm_model(
        x_shape=tuple(args.x_shape),
        axis=args.axis,
        epsilon=args.epsilon,
        output_file=args.output,
    )
