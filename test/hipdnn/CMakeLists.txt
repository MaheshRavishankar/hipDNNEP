# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# hipDNN test sources
target_sources(hipdnn_ep_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_conv.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/test_pointwise.cc
)

# Generate Conv test model
set(CONV_TEST_MODEL "${CMAKE_CURRENT_BINARY_DIR}/conv_test.onnx")
add_custom_command(
  OUTPUT ${CONV_TEST_MODEL} ${CMAKE_CURRENT_BINARY_DIR}/conv_test_weights.npy
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_conv_model.py
          --output ${CONV_TEST_MODEL}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_conv_model.py
  COMMENT "Generating Conv test model"
  VERBATIM
)
add_custom_target(generate_conv_test_model DEPENDS ${CONV_TEST_MODEL})
add_dependencies(hipdnn_ep_tests generate_conv_test_model)

target_compile_definitions(hipdnn_ep_tests PRIVATE
  CONV_TEST_MODEL_PATH="${CONV_TEST_MODEL}"
)

# Generate pointwise test models (Mul, Sub, Add, Div)
set(POINTWISE_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(MUL_TEST_MODEL "${POINTWISE_MODEL_DIR}/mul_test.onnx")
set(SUB_TEST_MODEL "${POINTWISE_MODEL_DIR}/sub_test.onnx")
set(ADD_TEST_MODEL "${POINTWISE_MODEL_DIR}/add_test.onnx")
set(DIV_TEST_MODEL "${POINTWISE_MODEL_DIR}/div_test.onnx")
add_custom_command(
  OUTPUT ${MUL_TEST_MODEL} ${SUB_TEST_MODEL} ${ADD_TEST_MODEL} ${DIV_TEST_MODEL}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_pointwise_model.py
          --output-dir ${POINTWISE_MODEL_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_pointwise_model.py
  COMMENT "Generating pointwise test models"
  VERBATIM
)
add_custom_target(generate_pointwise_test_models
  DEPENDS ${MUL_TEST_MODEL} ${SUB_TEST_MODEL} ${ADD_TEST_MODEL} ${DIV_TEST_MODEL}
)
add_dependencies(hipdnn_ep_tests generate_pointwise_test_models)

target_compile_definitions(hipdnn_ep_tests PRIVATE
  MUL_TEST_MODEL_PATH="${MUL_TEST_MODEL}"
  SUB_TEST_MODEL_PATH="${SUB_TEST_MODEL}"
  ADD_TEST_MODEL_PATH="${ADD_TEST_MODEL}"
  DIV_TEST_MODEL_PATH="${DIV_TEST_MODEL}"
)
