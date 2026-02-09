// RUN: %hipdnn-ep-opt --eliminate-empty-tensors --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --split-input-file %s | %FileCheck %s

// Test single conv: hipdnn.execute with tensor types → hipdnn.execute with memref types.
//
// CHECK-LABEL: func.func @single_conv(
//  CHECK-SAME:   %[[INPUT:.*]]: memref<1x3x32x32xf32>,
//  CHECK-SAME:   %[[WEIGHT:.*]]: memref<16x3x3x3xf32>)
//  CHECK-SAME:   -> memref<1x16x30x30xf32>
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<1x16x30x30xf32>
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[INPUT]], %[[WEIGHT]] : memref<1x3x32x32xf32>, memref<16x3x3x3xf32>)
//  CHECK-SAME:     outs(%[[ALLOC]] : memref<1x16x30x30xf32>)
//   CHECK-NOT:   ->
//       CHECK:   return %[[ALLOC]] : memref<1x16x30x30xf32>

func.func @single_conv(
    %arg0: tensor<1x3x32x32xf32>,
    %arg1: tensor<16x3x3x3xf32>)
    -> tensor<1x16x30x30xf32> {
  %empty = tensor.empty() : tensor<1x16x30x30xf32>
  %0 = hipdnn.execute graph("hipdnn_graph_0")
      ins(%arg0, %arg1 : tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>)
      outs(%empty : tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
  return %0 : tensor<1x16x30x30xf32>
}

// -----

// Test two sequential convs: each hipdnn.execute gets its own memref.alloc.
//
// CHECK-LABEL: func.func @two_sequential_convs(
//  CHECK-SAME:   %[[INPUT:.*]]: memref<1x3x32x32xf32>,
//  CHECK-SAME:   %[[W0:.*]]: memref<16x3x3x3xf32>,
//  CHECK-SAME:   %[[W1:.*]]: memref<32x16x3x3xf32>)
//  CHECK-SAME:   -> memref<1x32x28x28xf32>
//       CHECK:   %[[ALLOC0:.*]] = memref.alloc() {{.*}} : memref<1x16x30x30xf32>
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[INPUT]], %[[W0]] : memref<1x3x32x32xf32>, memref<16x3x3x3xf32>)
//  CHECK-SAME:     outs(%[[ALLOC0]] : memref<1x16x30x30xf32>)
//       CHECK:   %[[ALLOC1:.*]] = memref.alloc() {{.*}} : memref<1x32x28x28xf32>
//       CHECK:   hipdnn.execute graph("hipdnn_graph_1")
//  CHECK-SAME:     ins(%[[ALLOC0]], %[[W1]] : memref<1x16x30x30xf32>, memref<32x16x3x3xf32>)
//  CHECK-SAME:     outs(%[[ALLOC1]] : memref<1x32x28x28xf32>)
//       CHECK:   return %[[ALLOC1]] : memref<1x32x28x28xf32>

func.func @two_sequential_convs(
    %arg0: tensor<1x3x32x32xf32>,
    %arg1: tensor<16x3x3x3xf32>,
    %arg2: tensor<32x16x3x3xf32>)
    -> tensor<1x32x28x28xf32> {
  %empty0 = tensor.empty() : tensor<1x16x30x30xf32>
  %0 = hipdnn.execute graph("hipdnn_graph_0")
      ins(%arg0, %arg1 : tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>)
      outs(%empty0 : tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
  %empty1 = tensor.empty() : tensor<1x32x28x28xf32>
  %1 = hipdnn.execute graph("hipdnn_graph_1")
      ins(%0, %arg2 : tensor<1x16x30x30xf32>, tensor<32x16x3x3xf32>)
      outs(%empty1 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
  return %1 : tensor<1x32x28x28xf32>
}
