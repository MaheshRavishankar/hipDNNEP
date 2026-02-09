// RUN: %hipdnn-ep-opt --hipdnn-finalize-memrefs --split-input-file %s | %FileCheck %s

// Test single returned alloc: promoted to function argument, empty return.
//
// CHECK-LABEL: func.func @single_alloc(
//  CHECK-SAME:   %[[INPUT:.*]]: memref<2x3xf32>,
//  CHECK-SAME:   %[[WEIGHT:.*]]: memref<3x4xf32>,
//  CHECK-SAME:   %[[OUT:.*]]: memref<2x4xf32>)
//   CHECK-NOT:   ->
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[INPUT]], %[[WEIGHT]] : memref<2x3xf32>, memref<3x4xf32>)
//  CHECK-SAME:     outs(%[[OUT]] : memref<2x4xf32>)
//       CHECK:   return
//   CHECK-NOT:   memref

func.func @single_alloc(
    %arg0: memref<2x3xf32>,
    %arg1: memref<3x4xf32>) -> memref<2x4xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4xf32>
  hipdnn.execute graph("hipdnn_graph_0")
      ins(%arg0, %arg1 : memref<2x3xf32>, memref<3x4xf32>)
      outs(%alloc : memref<2x4xf32>)
  return %alloc : memref<2x4xf32>
}

// -----

// Test two sequential ops with one returned alloc: intermediate alloc stays,
// returned alloc is promoted.
//
// CHECK-LABEL: func.func @two_ops_one_returned(
//  CHECK-SAME:   %[[INPUT:.*]]: memref<1x3x32x32xf32>,
//  CHECK-SAME:   %[[W0:.*]]: memref<16x3x3x3xf32>,
//  CHECK-SAME:   %[[W1:.*]]: memref<32x16x3x3xf32>,
//  CHECK-SAME:   %[[OUT:.*]]: memref<1x32x28x28xf32>)
//   CHECK-NOT:   ->
//       CHECK:   %[[ALLOC0:.*]] = memref.alloc() {{.*}} : memref<1x16x30x30xf32>
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[INPUT]], %[[W0]] : memref<1x3x32x32xf32>, memref<16x3x3x3xf32>)
//  CHECK-SAME:     outs(%[[ALLOC0]] : memref<1x16x30x30xf32>)
//       CHECK:   hipdnn.execute graph("hipdnn_graph_1")
//  CHECK-SAME:     ins(%[[ALLOC0]], %[[W1]] : memref<1x16x30x30xf32>, memref<32x16x3x3xf32>)
//  CHECK-SAME:     outs(%[[OUT]] : memref<1x32x28x28xf32>)
//       CHECK:   return
//   CHECK-NOT:   memref

func.func @two_ops_one_returned(
    %arg0: memref<1x3x32x32xf32>,
    %arg1: memref<16x3x3x3xf32>,
    %arg2: memref<32x16x3x3xf32>) -> memref<1x32x28x28xf32> {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<1x16x30x30xf32>
  hipdnn.execute graph("hipdnn_graph_0")
      ins(%arg0, %arg1 : memref<1x3x32x32xf32>, memref<16x3x3x3xf32>)
      outs(%alloc0 : memref<1x16x30x30xf32>)
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<1x32x28x28xf32>
  hipdnn.execute graph("hipdnn_graph_1")
      ins(%alloc0, %arg2 : memref<1x16x30x30xf32>, memref<32x16x3x3xf32>)
      outs(%alloc1 : memref<1x32x28x28xf32>)
  return %alloc1 : memref<1x32x28x28xf32>
}

// -----

// Negative test: return value not from alloc — left unchanged.
//
// CHECK-LABEL: func.func @no_alloc_return(
//  CHECK-SAME:   %[[INPUT:.*]]: memref<2x3xf32>)
//  CHECK-SAME:   -> memref<2x3xf32>
//       CHECK:   return %[[INPUT]] : memref<2x3xf32>

func.func @no_alloc_return(
    %arg0: memref<2x3xf32>) -> memref<2x3xf32> {
  return %arg0 : memref<2x3xf32>
}
