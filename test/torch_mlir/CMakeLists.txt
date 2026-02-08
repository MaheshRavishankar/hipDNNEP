# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Skip if lit testing tools are not available
if(NOT EXISTS "${FILECHECK_PATH}" OR NOT LIT_COMMAND)
  return()
endif()

# Set paths to tools for lit config
set(HIPDNN_EP_OPT_PATH "${CMAKE_BINARY_DIR}/tools/hipdnn_ep_opt/hipdnn-ep-opt")
set(HIPDNN_ONNX_TO_TORCH_MLIR_PATH "${CMAKE_BINARY_DIR}/tools/hipdnn_onnx_to_torch_mlir/hipdnn-onnx-to-torch-mlir")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY
)

add_custom_target(check-torch-mlir
  COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR} -v
  DEPENDS hipdnn-onnx-to-torch-mlir hipdnn-ep-opt
  COMMENT "Running Torch-MLIR lit tests"
)

# Register each lit test file as a separate CTest entry.
set(LIT_TESTS
  passes/hipdnn_offload.mlir
  passes/hipdnn_graph_to_executable.mlir
  ep/gen_test_models.py
)

foreach(test_file ${LIT_TESTS})
  # Strip extension and convert path separators to :: for test name
  string(REGEX REPLACE "\\.[^.]+$" "" test_path "${test_file}")
  string(REPLACE "/" "::" test_name "${test_path}")

  add_test(NAME torch-mlir::${test_name}
    COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR} --filter "${test_file}"
  )
endforeach()
