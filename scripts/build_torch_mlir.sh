#!/bin/bash
# Script to build torch-mlir and LLVM for hipDNN EP
#
# Usage:
#   ./scripts/build_torch_mlir.sh [build_type]
#
# Arguments:
#   build_type: RelWithDebInfo (default), Debug, or Release
#
# This script builds LLVM/MLIR and torch-mlir from the submodule in
# third_party/torch-mlir and installs it to ../build/torch-mlir/install.
#
# The build is done in two stages:
#   1. Build and install LLVM/MLIR
#   2. Build torch-mlir as an out-of-tree project against the installed MLIR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="${1:-RelWithDebInfo}"
BASE_BUILD_DIR="${REPO_ROOT}/../build/torch-mlir"
LLVM_BUILD_DIR="${BASE_BUILD_DIR}/llvm-build"
TORCH_MLIR_BUILD_DIR="${BASE_BUILD_DIR}/torch-mlir-build"
INSTALL_DIR="${BASE_BUILD_DIR}/install"
TORCH_MLIR_SRC="${REPO_ROOT}/third_party/torch-mlir"
LLVM_SRC="${TORCH_MLIR_SRC}/externals/llvm-project"

echo "=============================================="
echo "Building torch-mlir (two-stage build)"
echo "=============================================="
echo "Build type:         ${BUILD_TYPE}"
echo "LLVM build dir:     ${LLVM_BUILD_DIR}"
echo "torch-mlir build:   ${TORCH_MLIR_BUILD_DIR}"
echo "Install dir:        ${INSTALL_DIR}"
echo "torch-mlir source:  ${TORCH_MLIR_SRC}"
echo "LLVM source:        ${LLVM_SRC}"
echo "=============================================="

# Check if submodules are initialized
if [ ! -d "${LLVM_SRC}/llvm" ]; then
    echo "Error: LLVM submodule not initialized."
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Create build directories
mkdir -p "${LLVM_BUILD_DIR}"
mkdir -p "${TORCH_MLIR_BUILD_DIR}"

#
# Stage 1: Build LLVM/MLIR
#
echo ""
echo "=============================================="
echo "Stage 1: Building LLVM/MLIR"
echo "=============================================="

cmake -G Ninja -B "${LLVM_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=OFF \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_INCLUDE_TESTS=ON \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    "${LLVM_SRC}/llvm"

echo ""
echo "Building LLVM/MLIR (this takes a while)..."
cmake --build "${LLVM_BUILD_DIR}" -- -j$(nproc)

echo ""
echo "Installing LLVM/MLIR..."
cmake --build "${LLVM_BUILD_DIR}" --target install

#
# Stage 2: Build torch-mlir (out-of-tree)
#
echo ""
echo "=============================================="
echo "Stage 2: Building torch-mlir"
echo "=============================================="

# Note: We would prefer to disable Python bindings (MLIR_ENABLE_BINDINGS_PYTHON=OFF)
# but torch-mlir's out-of-tree build mode unconditionally enables them at line 133
# of its CMakeLists.txt, so LLVM must be built with Python bindings enabled.
cmake -G Ninja -B "${TORCH_MLIR_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_DIR="${INSTALL_DIR}/lib/cmake/mlir" \
    -DLLVM_DIR="${INSTALL_DIR}/lib/cmake/llvm" \
    -DTORCH_MLIR_ENABLE_STABLEHLO=OFF \
    -DTORCH_MLIR_ENABLE_REFBACKEND=OFF \
    -DTORCH_MLIR_USE_INSTALLED_PYTORCH=OFF \
    -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF \
    -DTORCH_MLIR_OUT_OF_TREE_BUILD=ON \
    "${TORCH_MLIR_SRC}"

echo ""
echo "Building torch-mlir..."
cmake --build "${TORCH_MLIR_BUILD_DIR}" -- -j$(nproc)

echo ""
echo "Installing torch-mlir..."
cmake --build "${TORCH_MLIR_BUILD_DIR}" --target install

echo ""
echo "=============================================="
echo "torch-mlir build complete!"
echo "=============================================="
echo "Install directory: ${INSTALL_DIR}"
echo ""
echo "To build hipDNNEP with torch-mlir support:"
echo "  cmake --preset RelWithDebInfo-MLIR"
echo "  cmake --build --preset RelWithDebInfo-MLIR"
echo "=============================================="
