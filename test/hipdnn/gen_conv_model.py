#!/usr/bin/env python3
# Copyright (c) 2025, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate a simple Conv ONNX model for testing."""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_conv_model(
    batch=1,
    in_channels=1,
    out_channels=1,
    height=8,
    width=8,
    kernel_h=3,
    kernel_w=3,
    pad_h=1,
    pad_w=1,
    stride_h=1,
    stride_w=1,
    output_file="conv_test.onnx"
):
    """Create a simple Conv model."""

    # Input
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT,
                                       [batch, in_channels, height, width])

    # Weight (as initializer with random values)
    W_shape = [out_channels, in_channels, kernel_h, kernel_w]
    W_data = np.random.randn(*W_shape).astype(np.float32)
    W = helper.make_tensor('W', TensorProto.FLOAT, W_shape, W_data.flatten().tolist())

    # Output shape
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                       [batch, out_channels, out_h, out_w])

    # Conv node
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[kernel_h, kernel_w],
        pads=[pad_h, pad_w, pad_h, pad_w],
        strides=[stride_h, stride_w],
    )

    # Graph
    graph = helper.make_graph(
        [conv_node],
        'conv_test',
        [X],  # inputs
        [Y],  # outputs
        [W],  # initializers
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    print(f"Saved model to {output_file}")
    print(f"  Input shape: [{batch}, {in_channels}, {height}, {width}]")
    print(f"  Weight shape: {W_shape}")
    print(f"  Output shape: [{batch}, {out_channels}, {out_h}, {out_w}]")

    # Also save weights for reference comparison
    np.save(output_file.replace('.onnx', '_weights.npy'), W_data)
    print(f"Saved weights to {output_file.replace('.onnx', '_weights.npy')}")

    return model, W_data


def create_conv_bias_model(
    batch=1,
    in_channels=1,
    out_channels=1,
    height=8,
    width=8,
    kernel_h=3,
    kernel_w=3,
    pad_h=1,
    pad_w=1,
    stride_h=1,
    stride_w=1,
    scalar_bias=True,
    bias_value=0.5,
    output_file="conv_bias_test.onnx"
):
    """Create a Conv model with bias.

    When scalar_bias is True, the bias is a shape-[1] constant initializer
    that exercises the scalar embedding path.  Otherwise it is the standard
    shape-[out_channels] vector.
    """

    # Input
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT,
                                       [batch, in_channels, height, width])

    # Weight (as initializer with random values)
    W_shape = [out_channels, in_channels, kernel_h, kernel_w]
    W_data = np.random.randn(*W_shape).astype(np.float32)
    W = helper.make_tensor('W', TensorProto.FLOAT, W_shape, W_data.flatten().tolist())

    # Bias initializer
    if scalar_bias:
        B_shape = [1]
        B_data = np.array([bias_value], dtype=np.float32)
    else:
        B_shape = [out_channels]
        B_data = np.full(B_shape, bias_value, dtype=np.float32)
    B = helper.make_tensor('B', TensorProto.FLOAT, B_shape, B_data.flatten().tolist())

    # Output shape
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                       [batch, out_channels, out_h, out_w])

    # Conv node with bias
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W', 'B'],
        outputs=['Y'],
        kernel_shape=[kernel_h, kernel_w],
        pads=[pad_h, pad_w, pad_h, pad_w],
        strides=[stride_h, stride_w],
    )

    # Graph — X is a runtime input; W and B are initializers
    graph = helper.make_graph(
        [conv_node],
        'conv_bias_test',
        [X],          # inputs
        [Y],          # outputs
        [W, B],       # initializers
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    print(f"Saved model to {output_file}")
    print(f"  Input shape: [{batch}, {in_channels}, {height}, {width}]")
    print(f"  Weight shape: {W_shape}")
    print(f"  Bias shape: {B_shape}")
    print(f"  Output shape: [{batch}, {out_channels}, {out_h}, {out_w}]")

    return model


def create_conv_nhwc_model(
    batch=1,
    in_channels=1,
    out_channels=1,
    height=8,
    width=8,
    kernel_h=3,
    kernel_w=3,
    pad_h=1,
    pad_w=1,
    stride_h=1,
    stride_w=1,
    output_file="conv_nhwc_test.onnx"
):
    """Create a Conv model with channels_last=1 (NHWC layout).

    The input and output use NHWC shape ordering [N, H, W, C].
    The weight remains in NCHW order [K, C, kH, kW] as ORT's layout
    transformer does not transpose the filter.
    The Conv node carries a ``channels_last=1`` integer attribute so the
    EP detects NHWC layout.
    """

    # Input in NHWC order
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT,
                                       [batch, height, width, in_channels])

    # Weight stays NCHW [K, C, kH, kW]
    W_shape = [out_channels, in_channels, kernel_h, kernel_w]
    W_data = np.random.randn(*W_shape).astype(np.float32)
    W = helper.make_tensor('W', TensorProto.FLOAT, W_shape,
                           W_data.flatten().tolist())

    # Output in NHWC order
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                       [batch, out_h, out_w, out_channels])

    # Conv node with channels_last attribute
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=[kernel_h, kernel_w],
        pads=[pad_h, pad_w, pad_h, pad_w],
        strides=[stride_h, stride_w],
    )
    # Add channels_last=1 attribute (used by ORT's NhwcTransformer)
    conv_node.attribute.append(
        helper.make_attribute('channels_last', 1))

    # Graph
    graph = helper.make_graph(
        [conv_node],
        'conv_nhwc_test',
        [X],
        [Y],
        [W],
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    # Note: onnx.checker may warn about the non-standard channels_last
    # attribute, but the model is still valid for our EP.
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        pass  # channels_last is a custom attribute
    onnx.save(model, output_file)
    print(f"Saved NHWC model to {output_file}")
    print(f"  Input shape (NHWC): [{batch}, {height}, {width}, {in_channels}]")
    print(f"  Weight shape (NCHW): {W_shape}")
    print(f"  Output shape (NHWC): [{batch}, {out_h}, {out_w}, {out_channels}]")

    # Save weights for reference comparison
    np.save(output_file.replace('.onnx', '_weights.npy'), W_data)

    return model, W_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="conv_test.onnx")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    create_conv_model(
        batch=args.batch,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        height=args.height,
        width=args.width,
        kernel_h=args.kernel,
        kernel_w=args.kernel,
        pad_h=args.pad,
        pad_w=args.pad,
        stride_h=args.stride,
        stride_w=args.stride,
        output_file=args.output
    )

    # Also generate the conv+bias model
    create_conv_bias_model(
        batch=args.batch,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        height=args.height,
        width=args.width,
        kernel_h=args.kernel,
        kernel_w=args.kernel,
        pad_h=args.pad,
        pad_w=args.pad,
        stride_h=args.stride,
        stride_w=args.stride,
        output_file=args.output.replace('.onnx', '_bias.onnx')
    )
