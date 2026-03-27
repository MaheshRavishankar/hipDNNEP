#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate MultiHeadAttention (SDPA) ONNX models for testing."""

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_mha_model(
    batch_size=2,
    seq_len_q=4,
    seq_len_kv=4,
    num_heads=2,
    head_size=8,
    unidirectional=0,
    scale=0.0,
    output_file="mha_test.onnx",
):
    """Create a MultiHeadAttention model: output = SDPA(Q, K, V).

    Q shape: [B, S_q, num_heads * head_size]
    K shape: [B, S_kv, num_heads * head_size]
    V shape: [B, S_kv, num_heads * head_size]
    output shape: [B, S_q, num_heads * head_size]
    """
    hidden_size = num_heads * head_size

    Q = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT, [batch_size, seq_len_q, hidden_size]
    )
    K = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT, [batch_size, seq_len_kv, hidden_size]
    )
    V = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT, [batch_size, seq_len_kv, hidden_size]
    )
    Y = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len_q, hidden_size]
    )

    attrs = {"num_heads": num_heads}
    if unidirectional:
        attrs["unidirectional"] = 1
    if scale != 0.0:
        attrs["scale"] = scale

    node = helper.make_node(
        "MultiHeadAttention",
        inputs=["query", "key", "value"],
        outputs=["output"],
        domain="com.microsoft",
        **attrs,
    )

    graph = helper.make_graph(
        [node],
        "mha_test",
        [Q, K, V],
        [Y],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 8

    onnx.save(model, output_file)
    print(f"Saved MultiHeadAttention model to {output_file}")
    print(
        f"  B={batch_size}, S_q={seq_len_q}, S_kv={seq_len_kv}, "
        f"H={num_heads}, D={head_size}, unidirectional={unidirectional}, "
        f"scale={scale}"
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len-q", type=int, default=4)
    parser.add_argument("--seq-len-kv", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=8)
    parser.add_argument("--unidirectional", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--output", "-o", default="mha_test.onnx")
    args = parser.parse_args()

    create_mha_model(
        batch_size=args.batch_size,
        seq_len_q=args.seq_len_q,
        seq_len_kv=args.seq_len_kv,
        num_heads=args.num_heads,
        head_size=args.head_size,
        unidirectional=args.unidirectional,
        scale=args.scale,
        output_file=args.output,
    )
