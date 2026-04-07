#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate RotaryEmbedding (RoPE) ONNX models for testing.

Creates a com.microsoft.RotaryEmbedding contrib op model with pre-computed
cos/sin caches matching the sequence length (no gathering needed).
"""

import argparse
import math

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def compute_rope_caches(seq_len, head_size, base=10000.0):
    """Compute cos/sin caches for RoPE.

    Returns cos_cache and sin_cache of shape [seq_len, head_size // 2].
    """
    half = head_size // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
    positions = np.arange(seq_len, dtype=np.float32)
    # Outer product: [seq_len, half]
    freqs = np.outer(positions, inv_freq)
    cos_cache = np.cos(freqs).astype(np.float32)
    sin_cache = np.sin(freqs).astype(np.float32)
    return cos_cache, sin_cache


def create_rope_model(
    batch_size=2,
    num_heads=4,
    seq_len=16,
    head_size=64,
    output_file="rope_test.onnx",
):
    """Create a RotaryEmbedding model.

    input shape:        [B, num_heads, S, head_size]
    position_ids shape: [1, S]
    cos_cache shape:    [S, head_size/2]
    sin_cache shape:    [S, head_size/2]
    output shape:       [B, num_heads, S, head_size]
    """
    half = head_size // 2

    # Graph inputs
    X = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT,
        [batch_size, num_heads, seq_len, head_size],
    )
    # position_ids: [B, S] with sequential values 0..S-1
    pos_ids = helper.make_tensor_value_info(
        "position_ids", TensorProto.INT64, [batch_size, seq_len],
    )
    cos_cache = helper.make_tensor_value_info(
        "cos_cache", TensorProto.FLOAT, [seq_len, half],
    )
    sin_cache = helper.make_tensor_value_info(
        "sin_cache", TensorProto.FLOAT, [seq_len, half],
    )

    # Graph output
    Y = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT,
        [batch_size, num_heads, seq_len, head_size],
    )

    # RotaryEmbedding node (com.microsoft domain)
    rope_node = helper.make_node(
        "RotaryEmbedding",
        inputs=["input", "position_ids", "cos_cache", "sin_cache"],
        outputs=["output"],
        domain="com.microsoft",
        interleaved=0,
    )

    graph = helper.make_graph(
        [rope_node],
        "rope_test",
        [X, pos_ids, cos_cache, sin_cache],
        [Y],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 14),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 8

    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    print(f"Saved model to {output_file}")
    print(f"  Input shape: [{batch_size}, {num_heads}, {seq_len}, {head_size}]")
    print(f"  cos/sin cache shape: [{seq_len}, {half}]")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RoPE test model")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="rope_test.onnx")
    args = parser.parse_args()

    create_rope_model(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_size=args.head_size,
        output_file=args.output,
    )
