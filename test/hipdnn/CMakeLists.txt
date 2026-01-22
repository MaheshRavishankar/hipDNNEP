# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# hipDNN (Conv) test sources
target_sources(hipdnn_ep_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_conv.cc
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
