// RUN: %hipdnn-ep-opt --hipdnn-graph-to-executable --split-input-file %s | %FileCheck %s --check-prefix=CHECK

// Test conv2d graph conversion with single operation
// CHECK-LABEL: func.func @conv2d_simple
//       CHECK:   %[[RESULT:.*]] = torch.operator "hipdnn.executable"
//  CHECK-SAME:     {graph = @hipdnn_graph_0}
//   CHECK-NOT:   torch.operator "hipdnn.graph"
//       CHECK:   return %[[RESULT]]
//       CHECK: func.func private @hipdnn_graph_0(!torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>) -> !torch.vtensor<[1,16,30,30],f32>
func.func @conv2d_simple(%arg0: !torch.vtensor<[1,3,32,32],f32>, %arg1: !torch.vtensor<[16,3,3,3],f32>) -> !torch.vtensor<[1,16,30,30],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %output_padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.operator "hipdnn.graph"(%arg0, %arg1) : (!torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>) -> !torch.vtensor<[1,16,30,30],f32> {
  ^bb0(%input: !torch.vtensor<[1,3,32,32],f32>, %weight: !torch.vtensor<[16,3,3,3],f32>):
    %1 = torch.aten.convolution %input, %weight, %none, %stride, %padding, %dilation, %false, %output_padding, %int1 : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,16,30,30],f32>
    torch.operator_terminator %1 : !torch.vtensor<[1,16,30,30],f32>
  }
  return %0 : !torch.vtensor<[1,16,30,30],f32>
}

// -----

// Test that multiple conv2d graphs get sequential module-unique names
// CHECK-LABEL: func.func @multiple_conv2d
//       CHECK:   %[[R0:.*]] = torch.operator "hipdnn.executable"
//  CHECK-SAME:     {graph = @hipdnn_graph_0}
//       CHECK:   %[[R1:.*]] = torch.operator "hipdnn.executable"
//  CHECK-SAME:     {graph = @hipdnn_graph_1}
//       CHECK:   return %[[R1]]
//       CHECK: func.func private @hipdnn_graph_0(!torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>) -> !torch.vtensor<[1,16,30,30],f32>
//       CHECK: func.func private @hipdnn_graph_1(!torch.vtensor<[1,16,30,30],f32>, !torch.vtensor<[32,16,3,3],f32>) -> !torch.vtensor<[1,32,28,28],f32>
func.func @multiple_conv2d(%arg0: !torch.vtensor<[1,3,32,32],f32>, %arg1: !torch.vtensor<[16,3,3,3],f32>, %arg2: !torch.vtensor<[32,16,3,3],f32>) -> !torch.vtensor<[1,32,28,28],f32> {
  %none = torch.constant.none
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %output_padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  // First conv: 3->16 channels
  %0 = torch.operator "hipdnn.graph"(%arg0, %arg1) : (!torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>) -> !torch.vtensor<[1,16,30,30],f32> {
  ^bb0(%input: !torch.vtensor<[1,3,32,32],f32>, %weight: !torch.vtensor<[16,3,3,3],f32>):
    %1 = torch.aten.convolution %input, %weight, %none, %stride, %padding, %dilation, %false, %output_padding, %int1 : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[16,3,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,16,30,30],f32>
    torch.operator_terminator %1 : !torch.vtensor<[1,16,30,30],f32>
  }
  // Second conv: 16->32 channels
  %2 = torch.operator "hipdnn.graph"(%0, %arg2) : (!torch.vtensor<[1,16,30,30],f32>, !torch.vtensor<[32,16,3,3],f32>) -> !torch.vtensor<[1,32,28,28],f32> {
  ^bb0(%input2: !torch.vtensor<[1,16,30,30],f32>, %weight2: !torch.vtensor<[32,16,3,3],f32>):
    %3 = torch.aten.convolution %input2, %weight2, %none, %stride, %padding, %dilation, %false, %output_padding, %int1 : !torch.vtensor<[1,16,30,30],f32>, !torch.vtensor<[32,16,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,28,28],f32>
    torch.operator_terminator %3 : !torch.vtensor<[1,32,28,28],f32>
  }
  return %2 : !torch.vtensor<[1,32,28,28],f32>
}
