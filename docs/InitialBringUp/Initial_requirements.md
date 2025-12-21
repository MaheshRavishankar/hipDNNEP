### Project goals

As part of this project I want to build a prototype ONNXRuntime EP that is using hipDNN. To start with I want to support only convolutions, after which I will
1) extend it to other operations
2) support some fusion.

This is a prototype, but I want to make sure we use the best practices of building an EP. I am new to building EPs and this is a way for me to learn the best practices as well.

## Notes on hipDNN

hipDNN is a kernel library that is provided by AMD. This kernel library is available in TheRock. TheRock project is checked out in /home/mahesh/TheRock/TheRock. hipDNN is just a front end header only library that has multiple backends that actually provide the execution support for operations. For now there are two plugins
1. MIOpen
2. Fusilli (which is based on IREE)

For now I have built in /home/mahesh/TheRock/MaheshRelWithDebInfo hipDNN built with the Fusilli plugin enabled.

The initial goal is to build a EP that is built out of tree and can be used with ONNXRuntime. I have an empty github repo checked out in /home/mahesh/onnxruntime/hipDNNEP to build this out-of-tree EP.
