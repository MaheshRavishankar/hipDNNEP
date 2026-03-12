# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# hipDNN (Conv) test sources
target_sources(hipdnn_ep_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_conv.cc
)

# Generate Conv test models (plain and with bias)
set(CONV_TEST_MODEL "${CMAKE_CURRENT_BINARY_DIR}/conv_test.onnx")
set(CONV_BIAS_TEST_MODEL "${CMAKE_CURRENT_BINARY_DIR}/conv_test_bias.onnx")
add_custom_command(
  OUTPUT ${CONV_TEST_MODEL}
         ${CMAKE_CURRENT_BINARY_DIR}/conv_test_weights.npy
         ${CONV_BIAS_TEST_MODEL}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_conv_model.py
          --output ${CONV_TEST_MODEL}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_conv_model.py
  COMMENT "Generating Conv test models"
  VERBATIM
)
add_custom_target(generate_conv_test_model DEPENDS
  ${CONV_TEST_MODEL} ${CONV_BIAS_TEST_MODEL})
add_dependencies(hipdnn_ep_tests generate_conv_test_model)

target_compile_definitions(hipdnn_ep_tests PRIVATE
  CONV_TEST_MODEL_PATH="${CONV_TEST_MODEL}"
  CONV_BIAS_TEST_MODEL_PATH="${CONV_BIAS_TEST_MODEL}"
)

# hipDNN (Pointwise) test sources
target_sources(hipdnn_ep_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_pointwise.cc
)

# Generate Pointwise test models (Mul, Sub, Add, Div)
set(_pw_ops Mul Sub Add Div)
set(_pw_models "")
foreach(_op IN LISTS _pw_ops)
  string(TOLOWER "${_op}" _op_lower)
  set(_model "${CMAKE_CURRENT_BINARY_DIR}/${_op_lower}_test.onnx")
  add_custom_command(
    OUTPUT ${_model}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_pointwise_model.py
            --op ${_op} --output ${_model}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_pointwise_model.py
    COMMENT "Generating ${_op} test model"
    VERBATIM
  )
  list(APPEND _pw_models ${_model})
  string(TOUPPER "${_op}" _op_upper)
  target_compile_definitions(hipdnn_ep_tests PRIVATE
    ${_op_upper}_TEST_MODEL_PATH="${_model}"
  )
endforeach()

# Generate Pointwise scalar-input test models (B is scalar [1])
foreach(_op IN LISTS _pw_ops)
  string(TOLOWER "${_op}" _op_lower)
  set(_model "${CMAKE_CURRENT_BINARY_DIR}/${_op_lower}_scalar_test.onnx")
  add_custom_command(
    OUTPUT ${_model}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_pointwise_model.py
            --op ${_op} --a-shape 1 4 8 8 --b-shape 1 --output ${_model}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_pointwise_model.py
    COMMENT "Generating ${_op} scalar test model"
    VERBATIM
  )
  list(APPEND _pw_models ${_model})
  string(TOUPPER "${_op}" _op_upper)
  target_compile_definitions(hipdnn_ep_tests PRIVATE
    ${_op_upper}_SCALAR_TEST_MODEL_PATH="${_model}"
  )
endforeach()

add_custom_target(generate_pointwise_test_models DEPENDS ${_pw_models})
add_dependencies(hipdnn_ep_tests generate_pointwise_test_models)

# hipDNN (Unary Pointwise / Sigmoid) test sources
target_sources(hipdnn_ep_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_sigmoid.cc
)

# Generate Sigmoid test model
set(_unary_pw_ops Sigmoid)
set(_unary_pw_models "")
foreach(_op IN LISTS _unary_pw_ops)
  string(TOLOWER "${_op}" _op_lower)
  set(_model "${CMAKE_CURRENT_BINARY_DIR}/${_op_lower}_test.onnx")
  add_custom_command(
    OUTPUT ${_model}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_unary_pointwise_model.py
            --op ${_op} --output ${_model}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_unary_pointwise_model.py
    COMMENT "Generating ${_op} test model"
    VERBATIM
  )
  list(APPEND _unary_pw_models ${_model})
  string(TOUPPER "${_op}" _op_upper)
  target_compile_definitions(hipdnn_ep_tests PRIVATE
    ${_op_upper}_TEST_MODEL_PATH="${_model}"
  )
endforeach()

add_custom_target(generate_unary_pointwise_test_models DEPENDS ${_unary_pw_models})
add_dependencies(hipdnn_ep_tests generate_unary_pointwise_test_models)
