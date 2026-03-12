#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate ReduceSum / ReduceMax / ReduceMin ONNX models for testing."""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_reduction_model(
    op_type="ReduceSum",
    input_shape=(1, 4, 8, 8),
    axes=None,
    keepdims=True,
    noop_with_empty_axes=False,
    output_file="reduction_test.onnx",
    opset=18,
):
    """Create a reduction model: Y = op(X, axes=axes, keepdims=keepdims).

    For opset >= 13, axes are provided as a constant input tensor.
    For opset < 13, axes are an attribute (not supported here for simplicity).
    """
    # Compute output shape.
    x_np = np.empty(input_shape)
    if axes is None:
        if noop_with_empty_axes:
            y_shape = list(input_shape)
        else:
            y_shape = [1] * len(input_shape) if keepdims else []
    else:
        y_shape = list(input_shape)
        for a in sorted(axes):
            a_norm = a if a >= 0 else a + len(input_shape)
            y_shape[a_norm] = 1
        if not keepdims:
            # Remove reduced dimensions.
            reduced = set(a if a >= 0 else a + len(input_shape) for a in axes)
            y_shape = [s for i, s in enumerate(y_shape) if i not in reduced]

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(input_shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y_shape)

    inputs = ["X"]
    initializers = []
    graph_inputs = [X]

    if axes is not None:
        axes_np = np.array(axes, dtype=np.int64)
        axes_tensor = numpy_helper.from_array(axes_np, name="axes")
        initializers.append(axes_tensor)
        inputs.append("axes")
        axes_input = helper.make_tensor_value_info(
            "axes", TensorProto.INT64, list(axes_np.shape)
        )
        graph_inputs.append(axes_input)

    attrs = {}
    attrs["keepdims"] = 1 if keepdims else 0
    if noop_with_empty_axes:
        attrs["noop_with_empty_axes"] = 1

    node = helper.make_node(
        op_type,
        inputs=inputs,
        outputs=["Y"],
        **attrs,
    )

    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_test",
        graph_inputs,
        [Y],
        initializer=initializers,
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)]
    )
    model.ir_version = 9

    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    print(f"Saved {op_type} model to {output_file}")
    print(f"  X shape: {list(input_shape)}, Y shape: {y_shape}, axes: {axes}, keepdims: {keepdims}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op", choices=["ReduceSum", "ReduceMax", "ReduceMin"], default="ReduceSum"
    )
    parser.add_argument("--output", "-o", default="reduction_test.onnx")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[1, 4, 8, 8])
    parser.add_argument("--axes", type=int, nargs="+", default=None)
    parser.add_argument("--keepdims", type=int, default=1)
    parser.add_argument("--noop-with-empty-axes", action="store_true")
    args = parser.parse_args()

    create_reduction_model(
        op_type=args.op,
        input_shape=tuple(args.input_shape),
        axes=args.axes,
        keepdims=bool(args.keepdims),
        noop_with_empty_axes=args.noop_with_empty_axes,
        output_file=args.output,
    )
