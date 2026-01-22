#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate ONNX models for MatMul and Gemm testing."""

import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def create_matmul_model(m: int, k: int, n: int, dtype: str = "float32") -> onnx.ModelProto:
    """Create a simple MatMul model: Y = A @ B"""
    onnx_dtype = TensorProto.FLOAT if dtype == "float32" else TensorProto.FLOAT16

    # Create input/output value infos
    a = helper.make_tensor_value_info("A", onnx_dtype, [m, k])
    b = helper.make_tensor_value_info("B", onnx_dtype, [k, n])
    y = helper.make_tensor_value_info("Y", onnx_dtype, [m, n])

    # Create MatMul node
    matmul_node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"])

    # Create graph
    graph = helper.make_graph([matmul_node], "matmul_test", [a, b], [y])

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    return model


def create_gemm_model(
    m: int,
    k: int,
    n: int,
    trans_a: bool = False,
    trans_b: bool = False,
    alpha: float = 1.0,
    beta: float = 1.0,
    has_bias: bool = False,
    dtype: str = "float32",
) -> onnx.ModelProto:
    """Create a Gemm model: Y = alpha * op(A) @ op(B) + beta * C"""
    onnx_dtype = TensorProto.FLOAT if dtype == "float32" else TensorProto.FLOAT16

    # Input shapes depend on transpose flags
    a_shape = [k, m] if trans_a else [m, k]
    b_shape = [n, k] if trans_b else [k, n]

    # Create input/output value infos
    inputs = [
        helper.make_tensor_value_info("A", onnx_dtype, a_shape),
        helper.make_tensor_value_info("B", onnx_dtype, b_shape),
    ]

    gemm_inputs = ["A", "B"]

    if has_bias:
        inputs.append(helper.make_tensor_value_info("C", onnx_dtype, [m, n]))
        gemm_inputs.append("C")

    y = helper.make_tensor_value_info("Y", onnx_dtype, [m, n])

    # Create Gemm node
    gemm_node = helper.make_node(
        "Gemm",
        inputs=gemm_inputs,
        outputs=["Y"],
        transA=1 if trans_a else 0,
        transB=1 if trans_b else 0,
        alpha=alpha,
        beta=beta,
    )

    # Create graph
    graph = helper.make_graph([gemm_node], "gemm_test", inputs, [y])

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    return model


def main():
    parser = argparse.ArgumentParser(description="Generate ONNX MatMul/Gemm test models")
    parser.add_argument("--output-dir", default=".", help="Output directory for models")
    args = parser.parse_args()

    # Generate MatMul test model
    print("Generating matmul_test.onnx...")
    matmul_model = create_matmul_model(m=64, k=128, n=32)
    onnx.save(matmul_model, f"{args.output_dir}/matmul_test.onnx")

    # Generate basic Gemm test model (no bias)
    print("Generating gemm_test.onnx...")
    gemm_model = create_gemm_model(m=64, k=128, n=32)
    onnx.save(gemm_model, f"{args.output_dir}/gemm_test.onnx")

    # Generate Gemm with bias
    print("Generating gemm_bias_test.onnx...")
    gemm_bias_model = create_gemm_model(m=64, k=128, n=32, has_bias=True)
    onnx.save(gemm_bias_model, f"{args.output_dir}/gemm_bias_test.onnx")

    # Generate Gemm with transA
    print("Generating gemm_trans_a_test.onnx...")
    gemm_trans_a_model = create_gemm_model(m=64, k=128, n=32, trans_a=True)
    onnx.save(gemm_trans_a_model, f"{args.output_dir}/gemm_trans_a_test.onnx")

    # Generate Gemm with transB
    print("Generating gemm_trans_b_test.onnx...")
    gemm_trans_b_model = create_gemm_model(m=64, k=128, n=32, trans_b=True)
    onnx.save(gemm_trans_b_model, f"{args.output_dir}/gemm_trans_b_test.onnx")

    # Generate Gemm with scaling
    print("Generating gemm_scaled_test.onnx...")
    gemm_scaled_model = create_gemm_model(m=64, k=128, n=32, alpha=0.5, beta=0.25, has_bias=True)
    onnx.save(gemm_scaled_model, f"{args.output_dir}/gemm_scaled_test.onnx")

    print("Done!")


if __name__ == "__main__":
    main()
