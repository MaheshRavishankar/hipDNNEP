// RUN: %hipdnn-ep-opt --hipdnn-offload --split-input-file %s | %FileCheck %s

// CHECK-LABEL: func.func @matmul
//       CHECK:   %[[RESULT:.*]] = torch.operator "hipdnn.graph"(%arg0, %arg1)
//       CHECK:   ^bb0(%[[A:.*]]: !torch.vtensor<[2,3],f32>, %[[B:.*]]: !torch.vtensor<[3,4],f32>):
//       CHECK:     %[[MM:.*]] = torch.aten.matmul %[[A]], %[[B]]
//       CHECK:     torch.operator_terminator %[[MM]]
//       CHECK:   return %[[RESULT]]
func.func @matmul(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// -----

// CHECK-LABEL: func.func @mm
//       CHECK:   %[[RESULT:.*]] = torch.operator "hipdnn.graph"(%arg0, %arg1)
//       CHECK:   ^bb0(%[[A:.*]]: !torch.vtensor<[4,5],f32>, %[[B:.*]]: !torch.vtensor<[5,6],f32>):
//       CHECK:     %[[MM:.*]] = torch.aten.mm %[[A]], %[[B]]
//       CHECK:     torch.operator_terminator %[[MM]]
//       CHECK:   return %[[RESULT]]
func.func @mm(%arg0: !torch.vtensor<[4,5],f32>, %arg1: !torch.vtensor<[5,6],f32>) -> !torch.vtensor<[4,6],f32> {
  %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[4,5],f32>, !torch.vtensor<[5,6],f32> -> !torch.vtensor<[4,6],f32>
  return %0 : !torch.vtensor<[4,6],f32>
}

// -----

// CHECK-LABEL: func.func @addmm
//       CHECK:   %[[RESULT:.*]] = torch.operator "hipdnn.graph"(%arg0, %arg1, %arg2)
//  CHECK-SAME:     : (!torch.vtensor<[4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,4],f32>)
//  CHECK-SAME:     -> !torch.vtensor<[3,4],f32>
//       CHECK:   ^bb0(%[[BIAS:.*]]: !torch.vtensor<[4],f32>, %[[MAT1:.*]]: !torch.vtensor<[3,4],f32>, %[[MAT2:.*]]: !torch.vtensor<[4,4],f32>):
//       CHECK:     %[[ALPHA:.*]] = torch.constant.int 1
//       CHECK:     torch.aten.addmm %[[BIAS]], %[[MAT1]], %[[MAT2]], %[[ALPHA]], %[[ALPHA]]
//       CHECK:     torch.operator_terminator
//       CHECK:   return %[[RESULT]]
func.func @addmm(%arg0: !torch.vtensor<[4],f32>, %arg1: !torch.vtensor<[3,4],f32>, %arg2: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[3,4],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.addmm %arg0, %arg1, %arg2, %int1, %int1 : !torch.vtensor<[4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,4],f32>
  return %0 : !torch.vtensor<[3,4],f32>
}

// -----

// CHECK-LABEL: func.func @conv2d
//       CHECK:   %[[RESULT:.*]] = torch.operator "hipdnn.graph"(%arg0, %arg1, %arg2)
//  CHECK-SAME:     : (!torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>, !torch.vtensor<[16],f32>)
//  CHECK-SAME:     -> !torch.vtensor<[1,16,30,30],f32>
//       CHECK:   ^bb0(%[[INPUT:.*]]: !torch.vtensor<[1,3,32,32],f32>, %[[WEIGHT:.*]]: !torch.vtensor<[16,3,3,3],f32>, %[[BIAS:.*]]: !torch.vtensor<[16],f32>):
//       CHECK:     torch.constant.int
//       CHECK:     torch.prim.ListConstruct
//       CHECK:     torch.aten.conv2d
//       CHECK:     torch.operator_terminator
//       CHECK:   return %[[RESULT]]
func.func @conv2d(%arg0: !torch.vtensor<[1,3,32,32],f32>, %arg1: !torch.vtensor<[16,3,3,3],f32>, %arg2: !torch.vtensor<[16],f32>) -> !torch.vtensor<[1,16,30,30],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.conv2d %arg0, %arg1, %arg2, %stride, %padding, %dilation, %int1 : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,16,30,30],f32>
  return %0 : !torch.vtensor<[1,16,30,30],f32>
}

// -----

// Verify that non-supported ops are not outlined
// CHECK-LABEL: func.func @relu_not_outlined
//       CHECK:   torch.aten.relu
//   CHECK-NOT:   torch.operator "hipdnn.graph"
func.func @relu_not_outlined(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.relu %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}
