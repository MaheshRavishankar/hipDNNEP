# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# hipBLAS-LT (MatMul/Gemm) test sources
target_sources(hipdnn_ep_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul.cc
)

# Generate MatMul/Gemm test models
set(MATMUL_TEST_MODELS
  ${CMAKE_CURRENT_BINARY_DIR}/matmul_test.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/gemm_test.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/gemm_bias_test.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/gemm_trans_a_test.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/gemm_trans_b_test.onnx
  ${CMAKE_CURRENT_BINARY_DIR}/gemm_scaled_test.onnx
)

add_custom_command(
  OUTPUT ${MATMUL_TEST_MODELS}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_matmul_model.py
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_matmul_model.py
  COMMENT "Generating MatMul/Gemm test models"
  VERBATIM
)
add_custom_target(generate_matmul_test_models DEPENDS ${MATMUL_TEST_MODELS})
add_dependencies(hipdnn_ep_tests generate_matmul_test_models)

target_compile_definitions(hipdnn_ep_tests PRIVATE
  MATMUL_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/matmul_test.onnx"
  GEMM_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_test.onnx"
  GEMM_BIAS_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_bias_test.onnx"
  GEMM_TRANS_A_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_trans_a_test.onnx"
  GEMM_TRANS_B_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_trans_b_test.onnx"
  GEMM_SCALED_TEST_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/gemm_scaled_test.onnx"
)
