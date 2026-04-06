# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Static library for HipDNNGraph
set(LIB_NAME hipdnn_ep_src_hipdnn_graph)
set(HIPDNN_GRAPH_SOURCES
  hipdnn_graph.cc
  hipdnn_backend_shim.cc
)

add_library(${LIB_NAME} STATIC ${HIPDNN_GRAPH_SOURCES})
add_library(hipdnn_ep::src::hipdnn_graph ALIAS ${LIB_NAME})

target_include_directories(${LIB_NAME} PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIR}
)

# We intentionally do NOT link the hipdnn_frontend / hipdnn_backend CMake
# targets because hipdnn_frontend transitively links hipdnn_backend, which
# would add libhipdnn_backend.so as a DT_NEEDED dependency.  That pulls in
# libamd_comgr.so → Rock's libLLVM.so into the global symbol table and
# conflicts with torch-mlir's statically linked LLVM.
#
# Instead we add only the include directories so that the frontend
# header-only templates and backend type definitions are available at
# compile time.  The actual backend function symbols are provided by
# hipdnn_backend_shim.cc which uses dlopen(RTLD_LOCAL) at runtime.
target_include_directories(${LIB_NAME} SYSTEM PUBLIC
  $<TARGET_PROPERTY:hipdnn_frontend,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:hipdnn_backend,INTERFACE_INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:hipdnn_data_sdk,INTERFACE_INCLUDE_DIRECTORIES>
  ${NLOHMANN_JSON_INCLUDE_DIR}
  ${ROCM_INCLUDE_DIR}
)

target_compile_definitions(${LIB_NAME} PRIVATE
  HIPDNN_BACKEND_LIB_PATH="${THEROCK_DIST}/lib/libhipdnn_backend.so"
)

target_link_libraries(${LIB_NAME} PUBLIC
  hipdnn_ep::src::utils
  hip::host
  ${CMAKE_DL_LIBS}
)

# Add MLIR includes when Torch-MLIR is enabled (for MLIR-based Build method)
if(HIPDNN_EP_ENABLE_TORCH_MLIR)
  target_include_directories(${LIB_NAME} PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${TORCH_MLIR_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
  )
  # hipDNN frontend headers require RTTI (dynamic_cast, typeid in
  # EngineOverrideUtils.hpp and KnobWrapper.hpp), but MLIR libraries are
  # compiled without RTTI. Use -fno-rtti on the target but override it
  # for hipdnn_graph.cc which includes the frontend headers.
  target_compile_options(${LIB_NAME} PRIVATE -fno-rtti)
  set_source_files_properties(hipdnn_graph.cc PROPERTIES COMPILE_OPTIONS -frtti)
  # The shim also includes hipdnn_backend.h which transitively includes HIP
  # headers that may use RTTI.
  set_source_files_properties(hipdnn_backend_shim.cc PROPERTIES COMPILE_OPTIONS -frtti)
  target_compile_definitions(${LIB_NAME} PUBLIC HIPDNN_EP_HAS_TORCH_MLIR)
  target_link_libraries(${LIB_NAME} PUBLIC
    ${TORCH_MLIR_LIB_DIR}/libTorchMLIRTorchDialect.a
    MLIRIR
    LLVMSupport
  )
endif()
