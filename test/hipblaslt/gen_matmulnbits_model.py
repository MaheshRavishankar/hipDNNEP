#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate ONNX models for MatMulNBits testing.

MatMulNBits (com.microsoft) performs:
  Y = A @ dequantize(B)^T
where A is [M, K] float/fp16, B is [N, k_blocks, blob_size] packed uint8 (int4),
scales is [N, k_blocks], and optional zero_points is packed uint8.
"""

import argparse
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def quantize_to_int4(weights: np.ndarray, block_size: int):
    """Quantize a float32 [N, K] weight matrix to int4 format.

    Returns:
        b_packed: [N, k_blocks, blob_size] uint8 (two int4 values per byte)
        scales: [N, k_blocks] float32
        zero_points: [N, zp_bytes] uint8 (packed int4 zero points)
    """
    N, K = weights.shape
    k_blocks = (K + block_size - 1) // block_size
    blob_size = block_size // 2

    b_packed = np.zeros((N, k_blocks, blob_size), dtype=np.uint8)
    scales = np.zeros((N, k_blocks), dtype=np.float32)
    zp_packed = np.zeros((N, (k_blocks + 1) // 2), dtype=np.uint8)

    for n in range(N):
        for block in range(k_blocks):
            k_start = block * block_size
            k_end = min(k_start + block_size, K)
            block_vals = weights[n, k_start:k_end]

            # Compute scale and zero point for this block.
            vmin = float(block_vals.min())
            vmax = float(block_vals.max())

            # Symmetric quantization around midpoint (zp=8 for unsigned 4-bit).
            if vmax == vmin:
                scale = 1.0
            else:
                scale = (vmax - vmin) / 15.0
            zp = 8  # default zero point

            scales[n, block] = scale

            # Pack zero point.
            zp_idx = block
            zp_byte = zp_idx // 2
            if zp_idx % 2 == 0:
                zp_packed[n, zp_byte] |= (zp & 0x0F)
            else:
                zp_packed[n, zp_byte] |= ((zp & 0x0F) << 4)

            # Quantize and pack values.
            for i in range(k_end - k_start):
                val = block_vals[i]
                q = int(round(val / scale + zp))
                q = max(0, min(15, q))

                byte_idx = i // 2
                if i % 2 == 0:
                    b_packed[n, block, byte_idx] |= (q & 0x0F)
                else:
                    b_packed[n, block, byte_idx] |= ((q & 0x0F) << 4)

    return b_packed, scales, zp_packed


def dequantize_int4(b_packed, scales, zp_packed, K, N, block_size):
    """Dequantize int4 weights back to float32 [K, N] for reference."""
    k_blocks = (K + block_size - 1) // block_size
    blob_size = block_size // 2
    out = np.zeros((K, N), dtype=np.float32)

    for n in range(N):
        for block in range(k_blocks):
            scale = scales[n, block]
            zp_idx = block
            zp_byte = zp_idx // 2
            if zp_idx % 2 == 0:
                zp = int(zp_packed[n, zp_byte] & 0x0F)
            else:
                zp = int((zp_packed[n, zp_byte] >> 4) & 0x0F)

            for i in range(block_size):
                k = block * block_size + i
                if k >= K:
                    break
                byte_idx = i // 2
                packed = b_packed[n, block, byte_idx]
                if i % 2 == 0:
                    val = int(packed & 0x0F)
                else:
                    val = int((packed >> 4) & 0x0F)
                out[k, n] = float(val - zp) * scale

    return out


def create_matmulnbits_model(
    m: int, k: int, n: int, block_size: int = 32
) -> tuple:
    """Create a MatMulNBits model with random quantized weights.

    Returns (model, dequantized_weights) for reference testing.
    """
    # Generate random weights and quantize.
    np.random.seed(42)
    weights_f32 = np.random.randn(n, k).astype(np.float32) * 0.1

    b_packed, scales, zp_packed = quantize_to_int4(weights_f32, block_size)

    k_blocks = (k + block_size - 1) // block_size

    # Create inputs.
    a_input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])

    # Create initializers for B, scales, zero_points.
    b_init = numpy_helper.from_array(b_packed, name="B")
    scales_init = numpy_helper.from_array(scales, name="scales")
    zp_init = numpy_helper.from_array(zp_packed, name="zero_points")

    # Create MatMulNBits node.
    matmulnbits_node = helper.make_node(
        "MatMulNBits",
        inputs=["A", "B", "scales", "zero_points"],
        outputs=["Y"],
        domain="com.microsoft",
        K=k,
        N=n,
        bits=4,
        block_size=block_size,
    )

    graph = helper.make_graph(
        [matmulnbits_node],
        "matmulnbits_test",
        [a_input],
        [y_output],
        [b_init, scales_init, zp_init],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 9

    # Compute reference dequantized weights for testing.
    deq_weights = dequantize_int4(b_packed, scales, zp_packed, k, n, block_size)

    return model, deq_weights


def create_matmulnbits_no_zp_model(
    m: int, k: int, n: int, block_size: int = 32
) -> tuple:
    """Create a MatMulNBits model without zero_points (uses default zp=8)."""
    np.random.seed(42)
    weights_f32 = np.random.randn(n, k).astype(np.float32) * 0.1

    b_packed, scales, zp_packed = quantize_to_int4(weights_f32, block_size)

    k_blocks = (k + block_size - 1) // block_size

    a_input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])

    b_init = numpy_helper.from_array(b_packed, name="B")
    scales_init = numpy_helper.from_array(scales, name="scales")

    matmulnbits_node = helper.make_node(
        "MatMulNBits",
        inputs=["A", "B", "scales"],
        outputs=["Y"],
        domain="com.microsoft",
        K=k,
        N=n,
        bits=4,
        block_size=block_size,
    )

    graph = helper.make_graph(
        [matmulnbits_node],
        "matmulnbits_no_zp_test",
        [a_input],
        [y_output],
        [b_init, scales_init],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 9

    deq_weights = dequantize_int4(b_packed, scales, zp_packed, k, n, block_size)
    return model, deq_weights


def main():
    parser = argparse.ArgumentParser(description="Generate ONNX MatMulNBits test models")
    parser.add_argument("--output-dir", default=".", help="Output directory for models")
    args = parser.parse_args()

    # Basic MatMulNBits with zero_points: A[8, 64] @ dequant(B)[64, 32] = Y[8, 32]
    print("Generating matmulnbits_test.onnx...")
    model, _ = create_matmulnbits_model(m=8, k=64, n=32, block_size=32)
    onnx.save(model, f"{args.output_dir}/matmulnbits_test.onnx")

    # MatMulNBits without zero_points
    print("Generating matmulnbits_no_zp_test.onnx...")
    model, _ = create_matmulnbits_no_zp_model(m=8, k=64, n=32, block_size=32)
    onnx.save(model, f"{args.output_dir}/matmulnbits_no_zp_test.onnx")

    # Larger dimensions
    print("Generating matmulnbits_large_test.onnx...")
    model, _ = create_matmulnbits_model(m=16, k=128, n=64, block_size=32)
    onnx.save(model, f"{args.output_dir}/matmulnbits_large_test.onnx")

    print("Done!")


if __name__ == "__main__":
    main()
