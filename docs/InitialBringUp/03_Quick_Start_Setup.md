# Quick Start Setup Guide

## Prerequisites

### 1. ONNXRuntime Build

Already built at: `/home/mahesh/onnxruntime/onnxruntime/build/RelWithDebInfo/`

Required headers located at:
- `/home/mahesh/onnxruntime/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h`
- `/home/mahesh/onnxruntime/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h`
- `/home/mahesh/onnxruntime/onnxruntime/include/onnxruntime/core/session/onnxruntime_ep_c_api.h`

### 2. hipDNN Installation

Built at: `/home/mahesh/TheRock/build/MaheshRelWithDebInfo/artifacts/hipdnn_run_gfx1100/ml-libs/hipDNN/stage/`

CMake config files:
```
/home/mahesh/TheRock/build/MaheshRelWithDebInfo/artifacts/hipdnn_run_gfx1100/ml-libs/hipDNN/stage/lib/cmake/
├── hipdnn_backend/hipdnn_backendConfig.cmake
├── hipdnn_frontend/hipdnn_frontendConfig.cmake
├── hipdnn_sdk/hipdnn_sdkConfig.cmake
└── ...
```

---

## Project Setup

### 1. Initialize Repository

```bash
cd /home/mahesh/onnxruntime/hipDNNEP

# Create directory structure
mkdir -p include/hipdnn_ep/kernels
mkdir -p src/kernels
mkdir -p test/models
mkdir -p cmake
```

### 2. Create CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(hipdnn_ep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# HIP
find_package(hip REQUIRED)

# hipDNN
set(HIPDNN_ROOT "/home/mahesh/TheRock/build/MaheshRelWithDebInfo/artifacts/hipdnn_run_gfx1100/ml-libs/hipDNN/stage")
list(APPEND CMAKE_PREFIX_PATH "${HIPDNN_ROOT}/lib/cmake")

find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_backend CONFIG REQUIRED)

# ONNXRuntime headers
set(ONNXRUNTIME_INCLUDE_DIR "/home/mahesh/onnxruntime/onnxruntime/include")

# Library
add_library(hipdnn_ep SHARED
    src/ep_factory.cc
    src/ep.cc
    src/ep_allocator.cc
    src/ep_data_transfer.cc
    src/hipdnn_ep_exports.cc
    src/kernels/conv_kernel.cc
)

target_include_directories(hipdnn_ep
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(hipdnn_ep
    PRIVATE
        hipdnn_frontend
        hipdnn_backend
        hip::host
)

# Symbol visibility
target_compile_definitions(hipdnn_ep PRIVATE HIPDNN_EP_EXPORTS)

if(UNIX)
    target_compile_options(hipdnn_ep PRIVATE -fvisibility=hidden)
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
```

### 3. Create CMakePresets.json

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "RelWithDebInfo",
      "displayName": "RelWithDebInfo",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    },
    {
      "name": "Debug",
      "displayName": "Debug",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "RelWithDebInfo",
      "configurePreset": "RelWithDebInfo"
    },
    {
      "name": "Debug",
      "configurePreset": "Debug"
    }
  ]
}
```

---

## Minimal Skeleton Files

### include/hipdnn_ep/export.h

```cpp
#pragma once

#ifdef _WIN32
  #ifdef HIPDNN_EP_EXPORTS
    #define HIPDNN_EP_API __declspec(dllexport)
  #else
    #define HIPDNN_EP_API __declspec(dllimport)
  #endif
#else
  #define HIPDNN_EP_API __attribute__((visibility("default")))
#endif
```

### include/hipdnn_ep/ep_factory.h

```cpp
#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_ep_c_api.h>
#include <string>

namespace hipdnn_ep {

class HipDNNEpFactory {
public:
    HipDNNEpFactory(const char* registration_name,
                    const OrtApi* ort_api,
                    const OrtEpApi* ep_api,
                    const OrtLogger* logger);
    ~HipDNNEpFactory();

    // OrtEpFactory interface
    const char* GetName() const;
    const char* GetVendor() const;
    uint32_t GetVendorId() const;
    const char* GetVersion() const;

    OrtStatus* GetSupportedDevices(
        const OrtHardwareDevice* const* devices,
        size_t num_devices,
        OrtEpDevice** ep_devices,
        size_t max_ep_devices,
        size_t* num_ep_devices);

    OrtStatus* CreateEp(/* ... */);
    void ReleaseEp(OrtEp* ep);

private:
    std::string name_;
    const OrtApi* ort_api_;
    const OrtEpApi* ep_api_;
    const OrtLogger* logger_;
};

}  // namespace hipdnn_ep
```

### src/hipdnn_ep_exports.cc

```cpp
#include "hipdnn_ep/export.h"
#include "hipdnn_ep/ep_factory.h"
#include <onnxruntime_c_api.h>
#include <memory>

using namespace hipdnn_ep;

static std::unique_ptr<HipDNNEpFactory> g_factory;

extern "C" {

HIPDNN_EP_API OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories) {

    if (max_factories < 1) {
        *num_factories = 0;
        return nullptr;
    }

    const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
    const OrtEpApi* ep_api = ort_api->GetEpApi();

    g_factory = std::make_unique<HipDNNEpFactory>(
        registration_name, ort_api, ep_api, default_logger);

    // Cast to OrtEpFactory* (need to implement the interface properly)
    factories[0] = reinterpret_cast<OrtEpFactory*>(g_factory.get());
    *num_factories = 1;

    return nullptr;  // Success
}

HIPDNN_EP_API OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
    g_factory.reset();
    return nullptr;
}

}  // extern "C"
```

---

## Build Commands

```bash
cd /home/mahesh/onnxruntime/hipDNNEP

# Configure
cmake --preset RelWithDebInfo

# Build
cmake --build --preset RelWithDebInfo

# The library will be at:
# build/RelWithDebInfo/libhipdnn_ep.so
```

---

## Test Loading

Create a simple test to verify the EP loads:

```cpp
// test/test_load.cc
#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // Register the EP library
    const char* lib_path = "build/RelWithDebInfo/libhipdnn_ep.so";
    OrtStatus* status = Ort::GetApi().RegisterExecutionProviderLibrary(
        env, "HipDNN", lib_path);

    if (status) {
        std::cerr << "Failed to load EP: "
                  << Ort::GetApi().GetErrorMessage(status) << std::endl;
        Ort::GetApi().ReleaseStatus(status);
        return 1;
    }

    std::cout << "EP loaded successfully!" << std::endl;
    return 0;
}
```

---

## Key Reference Files

### Example Plugin EP (copy patterns from here)

```
/home/mahesh/onnxruntime/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/
├── ep_factory.h          # Factory pattern
├── ep_factory.cc         # Factory implementation
├── ep.h                  # EP interface
├── ep.cc                 # EP implementation
├── ep_allocator.h        # Memory allocator
├── ep_arena.h/cc         # Arena allocator
├── ep_data_transfer.h/cc # Data transfer
├── ep_stream_support.h   # Stream handling
└── example_plugin_ep.cc  # Exports
```

### hipDNN Samples

```
/home/mahesh/TheRock/TheRock/rocm-libraries/projects/hipdnn/samples/
├── ConvFprop.cpp         # Forward conv
├── ConvDgrad.cpp         # Backward data
├── ConvWgrad.cpp         # Backward filter
└── FusedConvFpropActiv.cpp  # Conv + ReLU fusion
```

---

## Environment Variables

Add to your shell profile or run before building:

```bash
# hipDNN paths
export HIPDNN_ROOT="/home/mahesh/TheRock/build/MaheshRelWithDebInfo/artifacts/hipdnn_run_gfx1100/ml-libs/hipDNN/stage"
export LD_LIBRARY_PATH="${HIPDNN_ROOT}/lib:${LD_LIBRARY_PATH}"

# For running tests with ONNXRuntime
export LD_LIBRARY_PATH="/home/mahesh/onnxruntime/onnxruntime/build/RelWithDebInfo:${LD_LIBRARY_PATH}"
```

---

## Next Steps

1. Copy the example_plugin_ep code as a starting template
2. Adapt to use hipDNN instead of the example implementation
3. Add Conv kernel using hipDNN graph API
4. Test with simple Conv model
