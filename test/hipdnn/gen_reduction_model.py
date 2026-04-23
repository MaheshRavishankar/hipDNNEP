#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate ONNX Reduce* test models (ReduceSum, ReduceMean, ReduceMax, etc.)."""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


# Reduce ops that have `axes` as an attribute in opset 13 (true for every
# Reduce* except ReduceSum, which moved `axes` to an input in opset 13).
REDUCE_OPS_AXES_ATTR = {
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceL1",
    "ReduceL2",
}

# Reduce ops whose `axes` must be passed as an input tensor in opset 13.
REDUCE_OPS_AXES_INPUT = {
    "ReduceSum",
}


def _output_shape(x_shape, axes, keepdims):
    """Compute keep-dims output shape of a Reduce* op."""
    rank = len(x_shape)
    is_reduced = [False] * rank
    if not axes:
        is_reduced = [True] * rank
    else:
        for a in axes:
            axis = a + rank if a < 0 else a
            if 0 <= axis < rank:
                is_reduced[axis] = True
    if keepdims:
        return [1 if r else d for r, d in zip(is_reduced, x_shape)]
    return [d for r, d in zip(is_reduced, x_shape) if not r]


def create_reduction_model(
    op_type,
    x_shape,
    axes,
    keepdims=1,
    output_file="reduction_test.onnx",
):
    """Create a Reduce* model: Y = ReduceXxx(X, axes, keepdims)."""
    y_shape = _output_shape(list(x_shape), list(axes) if axes else [], keepdims)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y_shape)

    initializers = []
    node_inputs = ["X"]
    attrs = {"keepdims": keepdims}

    if op_type in REDUCE_OPS_AXES_INPUT:
        axes_name = "axes"
        initializers.append(
            numpy_helper.from_array(
                np.asarray(axes, dtype=np.int64), name=axes_name
            )
        )
        node_inputs.append(axes_name)
    else:
        # Attribute form.
        attrs["axes"] = list(axes)

    node = helper.make_node(
        op_type,
        inputs=node_inputs,
        outputs=["Y"],
        **attrs,
    )

    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_test",
        [X],
        [Y],
        initializer=initializers,
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    print(f"Saved {op_type} model to {output_file}")
    print(f"  X shape: {list(x_shape)}, axes: {list(axes)}, "
          f"keepdims: {keepdims}, Y shape: {y_shape}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        choices=sorted(REDUCE_OPS_AXES_ATTR | REDUCE_OPS_AXES_INPUT),
        default="ReduceSum",
    )
    parser.add_argument("--output", "-o", default="reduction_test.onnx")
    parser.add_argument(
        "--x-shape", type=int, nargs="+", default=[2, 4, 8, 8]
    )
    parser.add_argument(
        "--axes", type=int, nargs="+", default=[2, 3]
    )
    parser.add_argument("--keepdims", type=int, default=1)
    args = parser.parse_args()

    create_reduction_model(
        op_type=args.op,
        x_shape=tuple(args.x_shape),
        axes=args.axes,
        keepdims=args.keepdims,
        output_file=args.output,
    )
