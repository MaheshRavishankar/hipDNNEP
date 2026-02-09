// RUN: %hipdnn-ep-opt --hipdnn-offload-pipeline --split-input-file %s | %FileCheck %s

// End-to-end test: torch.aten.matmul → hipdnn.execute with memref args.
//
// CHECK-LABEL: func.func @matmul(
//  CHECK-SAME:   %[[A:.*]]: memref<2x3xf32>,
//  CHECK-SAME:   %[[B:.*]]: memref<3x4xf32>,
//  CHECK-SAME:   %[[OUT:.*]]: memref<2x4xf32>)
//   CHECK-NOT:   ->
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[A]], %[[B]] : memref<2x3xf32>, memref<3x4xf32>)
//  CHECK-SAME:     outs(%[[OUT]] : memref<2x4xf32>)
//       CHECK:   return
//   CHECK-NOT:   memref

func.func @matmul(
    %arg0: !torch.vtensor<[2,3],f32>,
    %arg1: !torch.vtensor<[3,4],f32>)
    -> !torch.vtensor<[2,4],f32>
    attributes {torch.onnx_meta.opset_version = 14 : si64} {
  %0 = torch.aten.matmul %arg0, %arg1
      : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,4],f32>
      -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// End-to-end test: torch.aten.mm → hipdnn.execute with memref args.
//
// CHECK-LABEL: func.func @mm(
//  CHECK-SAME:   %[[A:.*]]: memref<4x5xf32>,
//  CHECK-SAME:   %[[B:.*]]: memref<5x6xf32>,
//  CHECK-SAME:   %[[OUT:.*]]: memref<4x6xf32>)
//   CHECK-NOT:   ->
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[A]], %[[B]] : memref<4x5xf32>, memref<5x6xf32>)
//  CHECK-SAME:     outs(%[[OUT]] : memref<4x6xf32>)
//       CHECK:   return
//   CHECK-NOT:   memref

func.func @mm(
    %arg0: !torch.vtensor<[4,5],f32>,
    %arg1: !torch.vtensor<[5,6],f32>)
    -> !torch.vtensor<[4,6],f32>
    attributes {torch.onnx_meta.opset_version = 14 : si64} {
  %0 = torch.aten.mm %arg0, %arg1
      : !torch.vtensor<[4,5],f32>, !torch.vtensor<[5,6],f32>
      -> !torch.vtensor<[4,6],f32>
  return %0 : !torch.vtensor<[4,6],f32>
}

// -----

// End-to-end test: torch.aten.addmm → hipdnn.execute with memref args.
//
// CHECK-LABEL: func.func @addmm(
//  CHECK-SAME:   %[[BIAS:.*]]: memref<4xf32>,
//  CHECK-SAME:   %[[MAT1:.*]]: memref<3x4xf32>,
//  CHECK-SAME:   %[[MAT2:.*]]: memref<4x4xf32>,
//  CHECK-SAME:   %[[OUT:.*]]: memref<3x4xf32>)
//   CHECK-NOT:   ->
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[BIAS]], %[[MAT1]], %[[MAT2]] : memref<4xf32>, memref<3x4xf32>, memref<4x4xf32>)
//  CHECK-SAME:     outs(%[[OUT]] : memref<3x4xf32>)
//       CHECK:   return
//   CHECK-NOT:   memref

func.func @addmm(
    %arg0: !torch.vtensor<[4],f32>,
    %arg1: !torch.vtensor<[3,4],f32>,
    %arg2: !torch.vtensor<[4,4],f32>)
    -> !torch.vtensor<[3,4],f32>
    attributes {torch.onnx_meta.opset_version = 14 : si64} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.addmm %arg0, %arg1, %arg2, %int1, %int1
      : !torch.vtensor<[4],f32>, !torch.vtensor<[3,4],f32>,
        !torch.vtensor<[4,4],f32>, !torch.int, !torch.int
      -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// End-to-end test: torch.aten.conv2d → hipdnn.execute with memref args.
//
// CHECK-LABEL: func.func @conv2d(
//  CHECK-SAME:   %[[INPUT:.*]]: memref<1x3x32x32xf32>,
//  CHECK-SAME:   %[[WEIGHT:.*]]: memref<16x3x3x3xf32>,
//  CHECK-SAME:   %[[BIAS:.*]]: memref<16xf32>,
//  CHECK-SAME:   %[[OUT:.*]]: memref<1x16x30x30xf32>)
//   CHECK-NOT:   ->
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[INPUT]], %[[WEIGHT]], %[[BIAS]] : memref<1x3x32x32xf32>, memref<16x3x3x3xf32>, memref<16xf32>)
//  CHECK-SAME:     outs(%[[OUT]] : memref<1x16x30x30xf32>)
//       CHECK:   return
//   CHECK-NOT:   memref

func.func @conv2d(
    %arg0: !torch.vtensor<[1,3,32,32],f32>,
    %arg1: !torch.vtensor<[16,3,3,3],f32>,
    %arg2: !torch.vtensor<[16],f32>)
    -> !torch.vtensor<[1,16,30,30],f32>
    attributes {torch.onnx_meta.opset_version = 14 : si64} {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %stride = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0
      : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.conv2d %arg0, %arg1, %arg2, %stride, %padding, %dilation, %int1
      : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>,
        !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>,
        !torch.list<int>, !torch.int
      -> !torch.vtensor<[1,16,30,30],f32>
  return %0 : !torch.vtensor<[1,16,30,30],f32>
}

// -----

// End-to-end test: two sequential torch.aten.conv2d ops. The intermediate
// alloc stays internal; only the final output is promoted to a function arg.
//
// CHECK-LABEL: func.func @two_convs(
//  CHECK-SAME:   %[[INPUT:.*]]: memref<1x3x32x32xf32>,
//  CHECK-SAME:   %[[W0:.*]]: memref<16x3x3x3xf32>,
//  CHECK-SAME:   %[[B0:.*]]: memref<16xf32>,
//  CHECK-SAME:   %[[W1:.*]]: memref<32x16x3x3xf32>,
//  CHECK-SAME:   %[[B1:.*]]: memref<32xf32>,
//  CHECK-SAME:   %[[OUT:.*]]: memref<1x32x28x28xf32>)
//   CHECK-NOT:   ->
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<1x16x30x30xf32>
//       CHECK:   hipdnn.execute graph("hipdnn_graph_0")
//  CHECK-SAME:     ins(%[[INPUT]], %[[W0]], %[[B0]] : memref<1x3x32x32xf32>, memref<16x3x3x3xf32>, memref<16xf32>)
//  CHECK-SAME:     outs(%[[ALLOC]] : memref<1x16x30x30xf32>)
//       CHECK:   hipdnn.execute graph("hipdnn_graph_1")
//  CHECK-SAME:     ins(%[[ALLOC]], %[[W1]], %[[B1]] : memref<1x16x30x30xf32>, memref<32x16x3x3xf32>, memref<32xf32>)
//  CHECK-SAME:     outs(%[[OUT]] : memref<1x32x28x28xf32>)
//       CHECK:   return
//   CHECK-NOT:   memref

func.func @two_convs(
    %arg0: !torch.vtensor<[1,3,32,32],f32>,
    %arg1: !torch.vtensor<[16,3,3,3],f32>,
    %arg2: !torch.vtensor<[16],f32>,
    %arg3: !torch.vtensor<[32,16,3,3],f32>,
    %arg4: !torch.vtensor<[32],f32>)
    -> !torch.vtensor<[1,32,28,28],f32>
    attributes {torch.onnx_meta.opset_version = 14 : si64} {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %stride = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0
      : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.conv2d %arg0, %arg1, %arg2, %stride, %padding, %dilation, %int1
      : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>,
        !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>,
        !torch.list<int>, !torch.int
      -> !torch.vtensor<[1,16,30,30],f32>
  %1 = torch.aten.conv2d %0, %arg3, %arg4, %stride, %padding, %dilation, %int1
      : !torch.vtensor<[1,16,30,30],f32>, !torch.vtensor<[32,16,3,3],f32>,
        !torch.vtensor<[32],f32>, !torch.list<int>, !torch.list<int>,
        !torch.list<int>, !torch.int
      -> !torch.vtensor<[1,32,28,28],f32>
  return %1 : !torch.vtensor<[1,32,28,28],f32>
}
