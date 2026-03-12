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

# hipDNN (Reduction) test sources
target_sources(hipdnn_ep_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/test_reduction.cc
)

# Generate Reduction test models
set(_reduction_models "")

# ReduceSum: all axes (default)
set(_model "${CMAKE_CURRENT_BINARY_DIR}/reduce_sum_all_axes.onnx")
add_custom_command(
  OUTPUT ${_model}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
          --op ReduceSum --input-shape 1 4 8 8 --output ${_model}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
  COMMENT "Generating ReduceSum all-axes test model"
  VERBATIM
)
list(APPEND _reduction_models ${_model})
target_compile_definitions(hipdnn_ep_tests PRIVATE
  REDUCE_SUM_ALL_AXES_MODEL_PATH="${_model}"
)

# ReduceSum: axis 1 (non-contiguous reduction)
set(_model "${CMAKE_CURRENT_BINARY_DIR}/reduce_sum_axis1.onnx")
add_custom_command(
  OUTPUT ${_model}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
          --op ReduceSum --input-shape 1 4 8 8 --axes 1 --output ${_model}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
  COMMENT "Generating ReduceSum axis-1 test model"
  VERBATIM
)
list(APPEND _reduction_models ${_model})
target_compile_definitions(hipdnn_ep_tests PRIVATE
  REDUCE_SUM_AXIS1_MODEL_PATH="${_model}"
)

# ReduceSum: last two axes (contiguous reduction)
set(_model "${CMAKE_CURRENT_BINARY_DIR}/reduce_sum_last_two.onnx")
add_custom_command(
  OUTPUT ${_model}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
          --op ReduceSum --input-shape 1 4 8 8 --axes 2 3 --output ${_model}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
  COMMENT "Generating ReduceSum last-two-axes test model"
  VERBATIM
)
list(APPEND _reduction_models ${_model})
target_compile_definitions(hipdnn_ep_tests PRIVATE
  REDUCE_SUM_LAST_TWO_MODEL_PATH="${_model}"
)

# ReduceSum: axis 0 (first axis, non-contiguous, batch=2)
set(_model "${CMAKE_CURRENT_BINARY_DIR}/reduce_sum_axis0.onnx")
add_custom_command(
  OUTPUT ${_model}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
          --op ReduceSum --input-shape 2 4 8 8 --axes 0 --output ${_model}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
  COMMENT "Generating ReduceSum axis-0 test model"
  VERBATIM
)
list(APPEND _reduction_models ${_model})
target_compile_definitions(hipdnn_ep_tests PRIVATE
  REDUCE_SUM_AXIS0_MODEL_PATH="${_model}"
)

# ReduceMax: axis 1
set(_model "${CMAKE_CURRENT_BINARY_DIR}/reduce_max_axis1.onnx")
add_custom_command(
  OUTPUT ${_model}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
          --op ReduceMax --input-shape 1 4 8 8 --axes 1 --output ${_model}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
  COMMENT "Generating ReduceMax axis-1 test model"
  VERBATIM
)
list(APPEND _reduction_models ${_model})
target_compile_definitions(hipdnn_ep_tests PRIVATE
  REDUCE_MAX_AXIS1_MODEL_PATH="${_model}"
)

# ReduceMin: axis 1
set(_model "${CMAKE_CURRENT_BINARY_DIR}/reduce_min_axis1.onnx")
add_custom_command(
  OUTPUT ${_model}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
          --op ReduceMin --input-shape 1 4 8 8 --axes 1 --output ${_model}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_reduction_model.py
  COMMENT "Generating ReduceMin axis-1 test model"
  VERBATIM
)
list(APPEND _reduction_models ${_model})
target_compile_definitions(hipdnn_ep_tests PRIVATE
  REDUCE_MIN_AXIS1_MODEL_PATH="${_model}"
)

add_custom_target(generate_reduction_test_models DEPENDS ${_reduction_models})
add_dependencies(hipdnn_ep_tests generate_reduction_test_models)
