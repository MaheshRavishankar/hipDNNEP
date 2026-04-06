#!/usr/bin/env python3
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

"""Generate GroupQueryAttention (GQA) ONNX models for testing."""

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    print("Please install onnx and numpy: pip install onnx numpy")
    exit(1)


def create_gqa_model(
    batch_size=2,
    seq_len=16,
    num_heads=8,
    kv_num_heads=2,
    head_size=64,
    output_file="gqa_test.onnx",
):
    """Create a GroupQueryAttention model: output = GQA(Q, K, V).

    com.microsoft.GroupQueryAttention requires 7-12 inputs:
      0: query    [B, S, num_heads * D]
      1: key      [B, S, kv_num_heads * D]  (optional)
      2: value    [B, S, kv_num_heads * D]  (optional)
      3: past_key                            (optional, empty)
      4: past_value                          (optional, empty)
      5: seqlens_k      [B] int32            (required)
      6: total_sequence_length  scalar int32  (required)

    Inputs 5-6 are baked in as constant initializers so that the test
    only needs to feed Q, K, V at runtime.
    """
    q_hidden = num_heads * head_size
    kv_hidden = kv_num_heads * head_size

    Q = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT, [batch_size, seq_len, q_hidden]
    )
    K = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden]
    )
    V = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT, [batch_size, seq_len, kv_hidden]
    )
    Y = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len, q_hidden]
    )

    # Metadata inputs required by the GQA schema (inputs 5-6).
    # These are listed as graph inputs AND initializers: ORT uses the
    # initializer value when they aren't explicitly fed at runtime.
    seqlens_k_data = np.full([batch_size], seq_len - 1, dtype=np.int32)
    seqlens_k_init = numpy_helper.from_array(seqlens_k_data, name="seqlens_k")
    seqlens_k_info = helper.make_tensor_value_info(
        "seqlens_k", TensorProto.INT32, [batch_size]
    )

    total_seqlen_data = np.array(seq_len, dtype=np.int32)
    total_seqlen_init = numpy_helper.from_array(
        total_seqlen_data, name="total_sequence_length"
    )
    total_seqlen_info = helper.make_tensor_value_info(
        "total_sequence_length", TensorProto.INT32, []
    )

    node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            "query",
            "key",
            "value",
            "",  # past_key (not used)
            "",  # past_value (not used)
            "seqlens_k",
            "total_sequence_length",
        ],
        outputs=["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
    )

    graph = helper.make_graph(
        [node],
        "gqa_test",
        [Q, K, V, seqlens_k_info, total_seqlen_info],
        [Y],
        initializer=[seqlens_k_init, total_seqlen_init],
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
        f"  B={batch_size}, S={seq_len}, "
        f"H_q={num_heads}, H_kv={kv_num_heads}, D={head_size}"
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--kv-num-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--output", "-o", default="gqa_test.onnx")
    args = parser.parse_args()

    create_gqa_model(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        kv_num_heads=args.kv_num_heads,
        head_size=args.head_size,
        output_file=args.output,
    )
