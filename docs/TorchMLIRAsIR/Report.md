# Using Torch-MLIR as Intermediate IR for hipDNN Execution Provider

## 1. Intent

This document explores the feasibility of using Torch-MLIR as an intermediate
representation (IR) for an ONNXRuntime Execution Provider (EP) that leverages
hipDNN for GPU-accelerated compute operations. Using an MLIR-based IR to drive
execution will allow us to perform full program optimizations and handle tasks
like memory planning more easily during the compilation phase. Some
challenges/constraints to keep in mind are:

1. We want to avoid designing a completely new IR for representing ONNX
   programs. There are many front-end IRs, and the effort to define and
   maintain a new IR as ML programs evolve takes away from the effort needed
   to make a particular model performant in execution. It is easier to start
   with something that is feature complete and work with that instead.

2. If we are using a compiler-based approach, at the compilation stage of
   ONNXRuntime, we will need to generate an artifact that can be used to
   execute the subgraph.

The following sections describe a straw-man approach to building a Torch-MLIR-based
solution. Please see the end of the document for some alternative
approaches, and a discussion of risks.

## 2. Compilation Pipeline

The compilation is illustrated using a two-layer MLP (multi-layer perceptron)
as the running example. Each layer performs a matrix multiply followed by a
bias add: `output = input @ weights + bias`.

### 2.1. hipDNN Offload

The Torch-MLIR program is analyzed to find operations that can be offloaded to
hipDNN. Once identified, this sequence of operations is moved into the region
of a `torch.operator`. The `torch.operator` is fairly free-form and just takes
a name attribute to describe the operation. At this point we can just use
`hipdnn.graph` as a name for all the operations.

**Note**: We could define an actual operation called `hipdnn.graph` instead of
piggy-backing on `torch.operator`, but there doesn't seem to be much advantage
of that apart from being able to verify certain properties of a `hipdnn.graph`.

This transformation is shown below where a simple two-layer MLP program is
converted to `hipdnn.graph` operations.

```mlir
// Input: Two-layer MLP using standard torch.aten ops
func.func @two_layer_mlp(
    %x: !torch.vtensor<[32,128],f32>,
    %W1: !torch.vtensor<[128,256],f32>,
    %b1: !torch.vtensor<[256],f32>,
    %W2: !torch.vtensor<[256,64],f32>,
    %b2: !torch.vtensor<[64],f32>
) -> !torch.vtensor<[32,64],f32> {
  %int1 = torch.constant.int 1

  // Layer 1: x @ W1 + b1
  %mm1 = torch.aten.mm %x, %W1 : !torch.vtensor<[32,128],f32>,
      !torch.vtensor<[128,256],f32> -> !torch.vtensor<[32,256],f32>
  %add1 = torch.aten.add.Tensor %mm1, %b1, %int1 : !torch.vtensor<[32,256],f32>,
      !torch.vtensor<[256],f32>, !torch.int -> !torch.vtensor<[32,256],f32>

  // Layer 2: layer1 @ W2 + b2
  %mm2 = torch.aten.mm %add1, %W2 : !torch.vtensor<[32,256],f32>,
      !torch.vtensor<[256,64],f32> -> !torch.vtensor<[32,64],f32>
  %add2 = torch.aten.add.Tensor %mm2, %b2, %int1 : !torch.vtensor<[32,64],f32>,
      !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[32,64],f32>

  return %add2 : !torch.vtensor<[32,64],f32>
}
```

```mlir
// Output: MLP with hipdnn.graph regions
func.func @two_layer_mlp_hipdnn(
    %x: !torch.vtensor<[32,128],f32>,
    %W1: !torch.vtensor<[128,256],f32>,
    %b1: !torch.vtensor<[256],f32>,
    %W2: !torch.vtensor<[256,64],f32>,
    %b2: !torch.vtensor<[64],f32>
) -> !torch.vtensor<[32,64],f32> {

  // Layer 1: hipdnn.graph for x @ W1 + b1
  %layer1 = torch.operator "hipdnn.graph"(%x, %W1, %b1) : (
    !torch.vtensor<[32,128],f32>,
    !torch.vtensor<[128,256],f32>,
    !torch.vtensor<[256],f32>
  ) -> !torch.vtensor<[32,256],f32> {
  ^bb0(%in: !torch.vtensor<[32,128],f32>,
       %weight: !torch.vtensor<[128,256],f32>,
       %bias: !torch.vtensor<[256],f32>):
    %int1 = torch.constant.int 1
    %mm = torch.aten.mm %in, %weight : !torch.vtensor<[32,128],f32>,
        !torch.vtensor<[128,256],f32> -> !torch.vtensor<[32,256],f32>
    %add = torch.aten.add.Tensor %mm, %bias, %int1 : !torch.vtensor<[32,256],f32>,
        !torch.vtensor<[256],f32>, !torch.int -> !torch.vtensor<[32,256],f32>
    torch.operator_terminator %add : !torch.vtensor<[32,256],f32>
  }

  // Layer 2: hipdnn.graph for layer1 @ W2 + b2
  %layer2 = torch.operator "hipdnn.graph"(%layer1, %W2, %b2) : (
    !torch.vtensor<[32,256],f32>,
    !torch.vtensor<[256,64],f32>,
    !torch.vtensor<[64],f32>
  ) -> !torch.vtensor<[32,64],f32> {
  ^bb0(%in: !torch.vtensor<[32,256],f32>,
       %weight: !torch.vtensor<[256,64],f32>,
       %bias: !torch.vtensor<[64],f32>):
    %int1 = torch.constant.int 1
    %mm = torch.aten.mm %in, %weight : !torch.vtensor<[32,256],f32>,
        !torch.vtensor<[256,64],f32> -> !torch.vtensor<[32,64],f32>
    %add = torch.aten.add.Tensor %mm, %bias, %int1 : !torch.vtensor<[32,64],f32>,
        !torch.vtensor<[64],f32>, !torch.int -> !torch.vtensor<[32,64],f32>
    torch.operator_terminator %add : !torch.vtensor<[32,64],f32>
  }

  return %layer2 : !torch.vtensor<[32,64],f32>
}
```

### 2.2. hipDNN Compilation

The next step is for the computation specified by the region of each
`hipdnn.graph` operation to be compiled using hipDNN. The compiled hipDNN graph
is stored in a kernel cache within the EP, which can be retrieved using a name.
In the IR this can be represented by:

1. Declaring a `func.func` that represents the signature of the compiled graph,
   with the name used to retrieve the compiled artifact from the cache being
   the symbol name used for the declaration.

2. The compiled `hipdnn.graph` operation can now be replaced with a
   `torch.operator` with the name `hipdnn.executable` that represents a
   compiled hipDNN graph and refers to the name given to the compiled graph.

`hipdnn.graph` is replaced by a reference to its compiled form.

```mlir
// Compiled graph declarations
func.func private @layer1_compiled(
    !torch.vtensor<[32,128],f32>,
    !torch.vtensor<[128,256],f32>,
    !torch.vtensor<[256],f32>
) -> !torch.vtensor<[32,256],f32>

func.func private @layer2_compiled(
    !torch.vtensor<[32,256],f32>,
    !torch.vtensor<[256,64],f32>,
    !torch.vtensor<[64],f32>
) -> !torch.vtensor<[32,64],f32>

// MLP using hipdnn.executable ops that reference compiled graphs
func.func @two_layer_mlp_hipdnn(
    %x: !torch.vtensor<[32,128],f32>,
    %W1: !torch.vtensor<[128,256],f32>,
    %b1: !torch.vtensor<[256],f32>,
    %W2: !torch.vtensor<[256,64],f32>,
    %b2: !torch.vtensor<[64],f32>
) -> !torch.vtensor<[32,64],f32> {

  %layer1 = torch.operator "hipdnn.executable"(%x, %W1, %b1) {
    graph = @layer1_compiled
  } : (!torch.vtensor<[32,128],f32>, !torch.vtensor<[128,256],f32>,
       !torch.vtensor<[256],f32>) -> !torch.vtensor<[32,256],f32>

  %layer2 = torch.operator "hipdnn.executable"(%layer1, %W2, %b2) {
    graph = @layer2_compiled
  } : (!torch.vtensor<[32,256],f32>, !torch.vtensor<[256,64],f32>,
       !torch.vtensor<[64],f32>) -> !torch.vtensor<[32,64],f32>

  return %layer2 : !torch.vtensor<[32,64],f32>
}
```

After this stage, our program should contain only
`hipdnn.executable`s.

### 2.3. Memory Planning

Now that we know the different hipDNN invocations needed for the program,
we will need to allocate memory for the intermediates needed within the
subgraph. In this proposal we are choosing to convert this function to `tensor`
types to leverage upstream MLIR bufferization. So first, the Torch types are
lowered to standard MLIR tensor types:

```mlir
func.func private @layer1_compiled(
    tensor<32x128xf32>, tensor<128x256xf32>, tensor<256xf32>
) -> tensor<32x256xf32>

func.func private @layer2_compiled(
    tensor<32x256xf32>, tensor<256x64xf32>, tensor<64xf32>
) -> tensor<32x64xf32>

func.func @two_layer_mlp_hipdnn(
    %x: tensor<32x128xf32>,
    %W1: tensor<128x256xf32>,
    %b1: tensor<256xf32>,
    %W2: tensor<256x64xf32>,
    %b2: tensor<64xf32>
) -> tensor<32x64xf32> {

  %layer1 = func.call @layer1_compiled(%x, %W1, %b1)
      : (tensor<32x128xf32>, tensor<128x256xf32>, tensor<256xf32>)
      -> tensor<32x256xf32>

  %layer2 = func.call @layer2_compiled(%layer1, %W2, %b2)
      : (tensor<32x256xf32>, tensor<256x64xf32>, tensor<64xf32>)
      -> tensor<32x64xf32>

  return %layer2 : tensor<32x64xf32>
}
```

Next we change this program to be destination-passing style, which means that
each result of the `func.call` is "tied" to one of its operands. In
other words, post bufferization the result is computed in the same buffer as
the buffer assigned to one of the operands. For the above function this would
mean that all the function calls get an extra argument that is the same shape
as the result with the attribute `bufferization.writable`.

```mlir
func.func private @layer1_compiled(
    tensor<32x128xf32>, tensor<128x256xf32>, tensor<256xf32>,
    tensor<32x256xf32> {bufferization.writable = true}
) -> tensor<32x256xf32>

func.func private @layer2_compiled(
    tensor<32x256xf32>, tensor<256x64xf32>, tensor<64xf32>,
    tensor<32x64xf32> {bufferization.writable = true}
) -> tensor<32x64xf32>

func.func @two_layer_mlp_hipdnn(
    %x: tensor<32x128xf32>,
    %W1: tensor<128x256xf32>,
    %b1: tensor<256xf32>,
    %W2: tensor<256x64xf32>,
    %b2: tensor<64xf32>,
    %out: tensor<32x64xf32> {bufferization.writable = true}
) -> tensor<32x64xf32> {

  %empty = tensor.empty() : tensor<32x256xf32>
  %layer1 = func.call @layer1_compiled(%x, %W1, %b1, %empty)
      : (tensor<32x128xf32>, tensor<128x256xf32>, tensor<256xf32>,
         tensor<32x256xf32>) -> tensor<32x256xf32>

  %layer2 = func.call @layer2_compiled(%layer1, %W2, %b2, %out)
      : (tensor<32x256xf32>, tensor<256x64xf32>, tensor<64xf32>,
         tensor<32x64xf32>) -> tensor<32x64xf32>

  return %layer2 : tensor<32x64xf32>
}
```

Notice that the intermediate here is allocated using a `tensor.empty`. For this
simple example we have no memory planning, but for more complex use cases we
can, at this stage, do the memory planning required for this subgraph. More
details of this can be fleshed out separately.

Post memory planning, we can leverage bufferization in MLIR to convert this
program into its `memref` representation as follows.

```mlir
func.func private @layer1_compiled(
    memref<32x128xf32>, memref<128x256xf32>, memref<256xf32>,
    memref<32x256xf32>)

func.func private @layer2_compiled(
    memref<32x256xf32>, memref<256x64xf32>, memref<64xf32>,
    memref<32x64xf32>)

func.func @two_layer_mlp_hipdnn(
    %x: memref<32x128xf32>,
    %W1: memref<128x256xf32>,
    %b1: memref<256xf32>,
    %W2: memref<256x64xf32>,
    %b2: memref<64xf32>,
    %out: memref<32x64xf32>
) {

  %layer1_out = memref.alloc() : memref<32x256xf32>
  func.call @layer1_compiled(%x, %W1, %b1, %layer1_out)
      : (memref<32x128xf32>, memref<128x256xf32>, memref<256xf32>,
         memref<32x256xf32>) -> ()

  func.call @layer2_compiled(%layer1_out, %W2, %b2, %out)
      : (memref<32x256xf32>, memref<256x64xf32>, memref<64xf32>,
         memref<32x64xf32>) -> ()

  return
}
```

Note that the `memref.alloc` here represents workspace memory needed to execute
the subgraph. The details of how this connects to hipDNN's workspace buffer
handling are left out for brevity.

At this stage, the compiled graph has symbols (`@layer1_compiled`,
`@layer2_compiled`) which are names of the hipDNN compiled graphs in the cache.
These aren't artifacts that can be linked into this program. So instead of
calling them we can replace this function with a call to a function called
`execute_hipdnn_graph`, making the original name of the function an argument to
the new function call. All the added declarations now become dead and can be
dropped.

```mlir
func.func @two_layer_mlp_hipdnn(
    %x: memref<32x128xf32>,
    %W1: memref<128x256xf32>,
    %b1: memref<256xf32>,
    %W2: memref<256x64xf32>,
    %b2: memref<64xf32>,
    %out: memref<32x64xf32>
) {

  %layer1_out = memref.alloc() : memref<32x256xf32>
  func.call @execute_hipdnn_graph("layer1_compiled", %x, %W1, %b1, %layer1_out)
      : (!llvm.ptr, memref<32x128xf32>, memref<128x256xf32>, memref<256xf32>,
         memref<32x256xf32>) -> ()

  func.call @execute_hipdnn_graph("layer2_compiled", %layer1_out, %W2, %b2, %out)
      : (!llvm.ptr, memref<32x256xf32>, memref<256x64xf32>, memref<64xf32>,
         memref<32x64xf32>) -> ()

  return
}
```

This function can now be compiled down to object code and linked with the
definition of `execute_hipdnn_graph`, which when called will look up the hipDNN
compiled graph from the cache and execute it using the operands passed to it.
This compiled + linked object file is the artifact produced by the compilation
phase to be used during execution phase.

## 3. Risks

1. **Creating larger subgraphs**: Ideally, we want to create larger subgraphs.
   Just using the operations that are natively supported by hipDNN would hamper
   creating subgraphs large enough to be able to be optimized and
   executed effectively, especially from the perspective of memory planning.
   One way around this is to have hipDNN get a new operation `hipdnn.custom_op`
   that takes any sequence of Torch-MLIR operations and creates a
   kernel (or sequence of kernels) for it. This is feasible by leveraging the
   Fusilli/IREE plugin which can compile any arbitrary Torch-MLIR program. This
   won't solve all the issues. We want to reduce the number of instances of
   `hipdnn.custom_op` in the program for efficient execution, but that becomes
   an optimization problem.

2. **Torch-MLIR for global optimizations**: The main risk is that we will be
   using Torch-MLIR for full program optimizations. This might not
   be the best suited dialect for this. The best suited dialect here would be
   upstream Tensor/Linalg dialects, where we can leverage the powerful
   transformation/canonicalizations that are in these dialects. This is still
   possible to use in the flow described here since we are still going through
   tensors/memrefs for the final compilation. We will have to see what is
   needed in terms of optimizations at this stage to decide if we want to build
   these optimizations in Torch-MLIR or leverage the more "standard" upstream
   dialects.

## 4. Alternatives

1. **Using ONNX-MLIR instead of Torch-MLIR**: ONNX-MLIR is also a front-end
   dialect, and potentially works better with ONNXRuntime. Our past experience
   with ONNX-MLIR has been poor. It is actually not centrally tied
   to ONNXRuntime, and seems to be unsupported at this point.

2. **Using IREE**: All of the issues that have been described above, except for
   offloading to hipDNN, are things that IREE is well equipped to handle. We
   can borrow a lot from the "know-how" of how IREE did this, but IREE cannot
   call hipDNN directly.
