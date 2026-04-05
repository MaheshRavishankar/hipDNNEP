#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate GroupQueryAttention (GQA) ONNX models for testing."""

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_gqa_model(
    batch_size=2,
    seq_len_q=16,
    seq_len_kv=16,
    num_heads=8,
    kv_num_heads=2,
    head_size=64,
    output_file="gqa_test.onnx",
):
    """Create a GroupQueryAttention model: output = GQA(Q, K, V).

    Q shape: [B, S_q, num_heads * head_size]
    K shape: [B, S_kv, kv_num_heads * head_size]
    V shape: [B, S_kv, kv_num_heads * head_size]
    output shape: [B, S_q, num_heads * head_size]
    """
    q_hidden = num_heads * head_size
    kv_hidden = kv_num_heads * head_size

    Q = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT, [batch_size, seq_len_q, q_hidden]
    )
    K = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT, [batch_size, seq_len_kv, kv_hidden]
    )
    V = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT, [batch_size, seq_len_kv, kv_hidden]
    )
    Y = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len_q, q_hidden]
    )

    node = helper.make_node(
        "GroupQueryAttention",
        inputs=["query", "key", "value"],
        outputs=["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
    )

    graph = helper.make_graph(
        [node],
        "gqa_test",
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
    print(f"Saved GroupQueryAttention model to {output_file}")
    print(
        f"  B={batch_size}, S_q={seq_len_q}, S_kv={seq_len_kv}, "
        f"H_q={num_heads}, H_kv={kv_num_heads}, D={head_size}"
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len-q", type=int, default=16)
    parser.add_argument("--seq-len-kv", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--kv-num-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--output", "-o", default="gqa_test.onnx")
    args = parser.parse_args()

    create_gqa_model(
        batch_size=args.batch_size,
        seq_len_q=args.seq_len_q,
        seq_len_kv=args.seq_len_kv,
        num_heads=args.num_heads,
        kv_num_heads=args.kv_num_heads,
        head_size=args.head_size,
        output_file=args.output,
    )
