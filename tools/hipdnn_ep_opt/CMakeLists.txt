# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

add_executable(hipdnn-ep-opt
  hipdnn-ep-opt.cc
)

target_compile_options(hipdnn-ep-opt PRIVATE -fno-rtti)

target_link_libraries(hipdnn-ep-opt PRIVATE
  hipdnn_ep_passes
  MLIROptLib
  MLIRSupport
)
