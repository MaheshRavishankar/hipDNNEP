# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

# Core EP sources
target_sources(hipdnn_ep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/ep.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ep_allocator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ep_data_transfer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ep_factory.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ep_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/hipdnn_ep_exports.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/node_compute_info.cc
)
