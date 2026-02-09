# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

set(LIB_NAME hipdnn_ep_src_hipdnn_dialect)
add_library(${LIB_NAME} STATIC
  HipDNNDialect.cpp
  BufferizableOpInterfaceImpl.cpp
)
add_library(hipdnn_ep::src::hipdnn_dialect ALIAS ${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/src/hipdnn_dialect
)

target_compile_options(${LIB_NAME} PRIVATE -fno-rtti)

target_link_libraries(${LIB_NAME} PUBLIC
  MLIRIR
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
)

add_dependencies(${LIB_NAME} HipDNNDialectIncGen)
