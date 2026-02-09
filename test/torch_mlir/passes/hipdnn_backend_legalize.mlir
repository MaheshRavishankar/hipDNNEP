// RUN: %hipdnn-ep-opt --hipdnn-backend-legalize --split-input-file %s | %FileCheck %s

// Test single conv: hipdnn.executable → hipdnn.execute with tensor.empty for DPS out.
//
// CHECK-LABEL: func.func @single_conv(
//  CHECK-SAME:   %[[INPUT:.*]]: tensor<1x3x32x32xf32>,
//  CHECK-SAME:   %[[WEIGHT:.*]]: tensor<16x3x3x3xf32>)
//  CHECK-SAME:   -> tensor<1x16x30x30xf32>
//       CHECK:   %[[EMPTY:.*]] = tensor.empty() : tensor<1x16x30x30xf32>
//       CHECK:   %[[RESULT:.*]] = hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[INPUT]], %[[WEIGHT]] : tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>)
//  CHECK-SAME:     outs(%[[EMPTY]] : tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
//       CHECK:   return %[[RESULT]] : tensor<1x16x30x30xf32>
//
//   CHECK-NOT: torch
//   CHECK-NOT: hipdnn.executable

func.func @single_conv(
    %arg0: !torch.vtensor<[1,3,32,32],f32>,
    %arg1: !torch.vtensor<[16,3,3,3],f32>)
    -> !torch.vtensor<[1,16,30,30],f32> {
  %0 = torch.operator "hipdnn.executable"(%arg0, %arg1)
      {graph = "hipdnn_graph_0"}
      : (!torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>)
      -> !torch.vtensor<[1,16,30,30],f32>
  return %0 : !torch.vtensor<[1,16,30,30],f32>
}

// -----

// Test two sequential convs: each hipdnn.executable gets its own hipdnn.execute and tensor.empty.
//
// CHECK-LABEL: func.func @two_sequential_convs(
//  CHECK-SAME:   %[[INPUT:.*]]: tensor<1x3x32x32xf32>,
//  CHECK-SAME:   %[[W0:.*]]: tensor<16x3x3x3xf32>,
//  CHECK-SAME:   %[[W1:.*]]: tensor<32x16x3x3xf32>)
//  CHECK-SAME:   -> tensor<1x32x28x28xf32>
//       CHECK:   %[[EMPTY0:.*]] = tensor.empty() : tensor<1x16x30x30xf32>
//       CHECK:   %[[R0:.*]] = hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[INPUT]], %[[W0]] : tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>)
//  CHECK-SAME:     outs(%[[EMPTY0]] : tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
//       CHECK:   %[[EMPTY1:.*]] = tensor.empty() : tensor<1x32x28x28xf32>
//       CHECK:   %[[R1:.*]] = hipdnn.execute graph("hipdnn_graph_1")
//  CHECK-SAME:     ins(%[[R0]], %[[W1]] : tensor<1x16x30x30xf32>, tensor<32x16x3x3xf32>)
//  CHECK-SAME:     outs(%[[EMPTY1]] : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
//       CHECK:   return %[[R1]] : tensor<1x32x28x28xf32>
//
//   CHECK-NOT: torch
//   CHECK-NOT: hipdnn.executable

func.func @two_sequential_convs(
    %arg0: !torch.vtensor<[1,3,32,32],f32>,
    %arg1: !torch.vtensor<[16,3,3,3],f32>,
    %arg2: !torch.vtensor<[32,16,3,3],f32>)
    -> !torch.vtensor<[1,32,28,28],f32> {
  %0 = torch.operator "hipdnn.executable"(%arg0, %arg1)
      {graph = "hipdnn_graph_0"}
      : (!torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>)
      -> !torch.vtensor<[1,16,30,30],f32>
  %1 = torch.operator "hipdnn.executable"(%0, %arg2)
      {graph = "hipdnn_graph_1"}
      : (!torch.vtensor<[1,16,30,30],f32>, !torch.vtensor<[32,16,3,3],f32>)
      -> !torch.vtensor<[1,32,28,28],f32>
  return %1 : !torch.vtensor<[1,32,28,28],f32>
}
