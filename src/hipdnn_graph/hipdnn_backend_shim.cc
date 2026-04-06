// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

// Shim for libhipdnn_backend.so — loads the real library via
// dlopen(RTLD_LOCAL) instead of direct linking.  RTLD_LOCAL keeps
// libamd_comgr.so (and its transitive libLLVM.so) out of the global
// symbol table, avoiding ABI conflicts with torch-mlir's statically
// linked LLVM.
//
// Every function exported by hipdnn_backend.h is defined here as a thin
// forwarder that resolves the real symbol via dlsym on first call.

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// Include backend header for types (hipdnnStatus_t, hipdnnHandle_t, etc.)
// We define HIPDNN_BACKEND_STATIC_DEFINE so that HIPDNN_BACKEND_EXPORT
// expands to nothing — the header becomes pure type declarations without
// dllimport/export attributes that would conflict with our definitions.
#define HIPDNN_BACKEND_STATIC_DEFINE
#include <hipdnn_backend.h>

namespace {

std::once_flag g_init_flag;
void* g_lib = nullptr;

void initLibrary() {
  g_lib = dlopen("libhipdnn_backend.so", RTLD_NOW | RTLD_LOCAL);
  if (!g_lib) {
    // Not fatal — the tool (hipdnn-ep-opt) may not need the runtime.
    // Calls to backend functions will return error status.
    fprintf(stderr,
            "hipdnn_backend_shim: dlopen(libhipdnn_backend.so) failed: %s\n",
            dlerror());
  }
}

void ensureLoaded() { std::call_once(g_init_flag, initLibrary); }

template <typename FnPtr>
FnPtr resolve(const char* name) {
  ensureLoaded();
  if (!g_lib) return nullptr;
  // Clear any prior error.
  dlerror();
  void* sym = dlsym(g_lib, name);
  if (!sym) {
    fprintf(stderr, "hipdnn_backend_shim: dlsym(%s) failed: %s\n", name,
            dlerror());
  }
  return reinterpret_cast<FnPtr>(sym);
}

// Helper: resolve once via a function-local static.
#define RESOLVE(name, type) \
  static auto fn = resolve<type>(#name); \
  if (!fn) return HIPDNN_STATUS_INTERNAL_ERROR;

#define RESOLVE_VOID(name, type) \
  static auto fn = resolve<type>(#name); \
  if (!fn) return;

#define RESOLVE_PTR(name, type) \
  static auto fn = resolve<type>(#name); \
  if (!fn) return nullptr;

}  // namespace

// ---------------------------------------------------------------------------
// Function shims — one per HIPDNN_BACKEND_EXPORT entry in hipdnn_backend.h
// ---------------------------------------------------------------------------

extern "C" {

hipdnnStatus_t hipdnnCreate(hipdnnHandle_t* handle) {
  using Fn = hipdnnStatus_t (*)(hipdnnHandle_t*);
  RESOLVE(hipdnnCreate, Fn);
  return fn(handle);
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle) {
  using Fn = hipdnnStatus_t (*)(hipdnnHandle_t);
  RESOLVE(hipdnnDestroy, Fn);
  return fn(handle);
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipStream_t streamId) {
  using Fn = hipdnnStatus_t (*)(hipdnnHandle_t, hipStream_t);
  RESOLVE(hipdnnSetStream, Fn);
  return fn(handle, streamId);
}

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipStream_t* streamId) {
  using Fn = hipdnnStatus_t (*)(hipdnnHandle_t, hipStream_t*);
  RESOLVE(hipdnnGetStream, Fn);
  return fn(handle, streamId);
}

hipdnnStatus_t hipdnnBackendCreateDescriptor(
    hipdnnBackendDescriptorType_t descriptorType,
    hipdnnBackendDescriptor_t* descriptor) {
  using Fn = hipdnnStatus_t (*)(hipdnnBackendDescriptorType_t,
                                hipdnnBackendDescriptor_t*);
  RESOLVE(hipdnnBackendCreateDescriptor, Fn);
  return fn(descriptorType, descriptor);
}

hipdnnStatus_t hipdnnBackendDestroyDescriptor(
    hipdnnBackendDescriptor_t descriptor) {
  using Fn = hipdnnStatus_t (*)(hipdnnBackendDescriptor_t);
  RESOLVE(hipdnnBackendDestroyDescriptor, Fn);
  return fn(descriptor);
}

hipdnnStatus_t hipdnnBackendExecute(hipdnnHandle_t handle,
                                    hipdnnBackendDescriptor_t executionPlan,
                                    hipdnnBackendDescriptor_t variantPack) {
  using Fn = hipdnnStatus_t (*)(hipdnnHandle_t, hipdnnBackendDescriptor_t,
                                hipdnnBackendDescriptor_t);
  RESOLVE(hipdnnBackendExecute, Fn);
  return fn(handle, executionPlan, variantPack);
}

hipdnnStatus_t hipdnnBackendFinalize(hipdnnBackendDescriptor_t descriptor) {
  using Fn = hipdnnStatus_t (*)(hipdnnBackendDescriptor_t);
  RESOLVE(hipdnnBackendFinalize, Fn);
  return fn(descriptor);
}

hipdnnStatus_t hipdnnBackendGetAttribute(
    hipdnnBackendDescriptor_t descriptor,
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType, int64_t requestedElementCount,
    int64_t* elementCount, void* arrayOfElements) {
  using Fn =
      hipdnnStatus_t (*)(hipdnnBackendDescriptor_t,
                         hipdnnBackendAttributeName_t,
                         hipdnnBackendAttributeType_t, int64_t, int64_t*,
                         void*);
  RESOLVE(hipdnnBackendGetAttribute, Fn);
  return fn(descriptor, attributeName, attributeType, requestedElementCount,
            elementCount, arrayOfElements);
}

hipdnnStatus_t hipdnnBackendSetAttribute(
    hipdnnBackendDescriptor_t descriptor,
    hipdnnBackendAttributeName_t attributeName,
    hipdnnBackendAttributeType_t attributeType, int64_t elementCount,
    const void* arrayOfElements) {
  using Fn = hipdnnStatus_t (*)(hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t, int64_t,
                                const void*);
  RESOLVE(hipdnnBackendSetAttribute, Fn);
  return fn(descriptor, attributeName, attributeType, elementCount,
            arrayOfElements);
}

const char* hipdnnGetErrorString(hipdnnStatus_t status) {
  using Fn = const char* (*)(hipdnnStatus_t);
  RESOLVE_PTR(hipdnnGetErrorString, Fn);
  return fn(status);
}

void hipdnnGetLastErrorString(char* message, size_t maxSize) {
  using Fn = void (*)(char*, size_t);
  RESOLVE_VOID(hipdnnGetLastErrorString, Fn);
  fn(message, maxSize);
}

void hipdnnPeekLastErrorString_ext(char* message, size_t maxSize) {
  using Fn = void (*)(char*, size_t);
  RESOLVE_VOID(hipdnnPeekLastErrorString_ext, Fn);
  fn(message, maxSize);
}

hipdnnStatus_t hipdnnBackendCreateAndDeserializeGraph_ext(
    hipdnnBackendDescriptor_t* descriptor, const uint8_t* serializedGraph,
    size_t graphByteSize) {
  using Fn = hipdnnStatus_t (*)(hipdnnBackendDescriptor_t*, const uint8_t*,
                                size_t);
  RESOLVE(hipdnnBackendCreateAndDeserializeGraph_ext, Fn);
  return fn(descriptor, serializedGraph, graphByteSize);
}

hipdnnStatus_t hipdnnBackendGetSerializedBinaryGraph_ext(
    hipdnnBackendDescriptor_t descriptor, size_t requestedByteSize,
    size_t* graphByteSize, uint8_t* serializedGraph) {
  using Fn = hipdnnStatus_t (*)(hipdnnBackendDescriptor_t, size_t, size_t*,
                                uint8_t*);
  RESOLVE(hipdnnBackendGetSerializedBinaryGraph_ext, Fn);
  return fn(descriptor, requestedByteSize, graphByteSize, serializedGraph);
}

void hipdnnLoggingCallback_ext(hipdnnSeverity_t severity, const char* msg) {
  using Fn = void (*)(hipdnnSeverity_t, const char*);
  RESOLVE_VOID(hipdnnLoggingCallback_ext, Fn);
  fn(severity, msg);
}

hipdnnStatus_t hipdnnSetEnginePluginPaths_ext(
    size_t numPaths, const char* const* pluginPaths,
    hipdnnPluginLoadingMode_ext_t loadingMode) {
  using Fn =
      hipdnnStatus_t (*)(size_t, const char* const*,
                         hipdnnPluginLoadingMode_ext_t);
  RESOLVE(hipdnnSetEnginePluginPaths_ext, Fn);
  return fn(numPaths, pluginPaths, loadingMode);
}

hipdnnStatus_t hipdnnSetPluginUnloadMode_ext(
    hipdnnPluginUnloadingMode_ext_t unloadingMode) {
  using Fn = hipdnnStatus_t (*)(hipdnnPluginUnloadingMode_ext_t);
  RESOLVE(hipdnnSetPluginUnloadMode_ext, Fn);
  return fn(unloadingMode);
}

hipdnnStatus_t hipdnnGetLoadedEnginePluginPaths_ext(hipdnnHandle_t handle,
                                                    size_t* numPluginPaths,
                                                    char** pluginPaths,
                                                    size_t* maxStringLen) {
  using Fn = hipdnnStatus_t (*)(hipdnnHandle_t, size_t*, char**, size_t*);
  RESOLVE(hipdnnGetLoadedEnginePluginPaths_ext, Fn);
  return fn(handle, numPluginPaths, pluginPaths, maxStringLen);
}

hipdnnStatus_t hipdnnSetUserLogCallback_ext(
    hipdnnUserLogCallback_t callback, hipdnnSeverity_t minLevel,
    hipdnnLogCallbackMode_t mode,
    hipdnnUserLogCallbackHandle_t userHandle) {
  using Fn = hipdnnStatus_t (*)(hipdnnUserLogCallback_t, hipdnnSeverity_t,
                                hipdnnLogCallbackMode_t,
                                hipdnnUserLogCallbackHandle_t);
  RESOLVE(hipdnnSetUserLogCallback_ext, Fn);
  return fn(callback, minLevel, mode, userHandle);
}

hipdnnStatus_t hipdnnBackendSetGlobalLogLevel_ext(hipdnnSeverity_t level) {
  using Fn = hipdnnStatus_t (*)(hipdnnSeverity_t);
  RESOLVE(hipdnnBackendSetGlobalLogLevel_ext, Fn);
  return fn(level);
}

hipdnnStatus_t hipdnnBackendGetGlobalLogLevel_ext(hipdnnSeverity_t* level) {
  using Fn = hipdnnStatus_t (*)(hipdnnSeverity_t*);
  RESOLVE(hipdnnBackendGetGlobalLogLevel_ext, Fn);
  return fn(level);
}

hipdnnStatus_t hipdnnGetEngineCount_ext(hipdnnHandle_t handle,
                                        size_t* numEngines) {
  using Fn = hipdnnStatus_t (*)(hipdnnHandle_t, size_t*);
  RESOLVE(hipdnnGetEngineCount_ext, Fn);
  return fn(handle, numEngines);
}

hipdnnStatus_t hipdnnGetEngineInfo_ext(hipdnnHandle_t handle,
                                       size_t engineIndex, int64_t* engineId,
                                       char* engineName,
                                       size_t* engineNameLen,
                                       char* pluginName,
                                       size_t* pluginNameLen, char* version,
                                       size_t* versionLen, char* type,
                                       size_t* typeLen) {
  using Fn = hipdnnStatus_t (*)(hipdnnHandle_t, size_t, int64_t*, char*,
                                size_t*, char*, size_t*, char*, size_t*, char*,
                                size_t*);
  RESOLVE(hipdnnGetEngineInfo_ext, Fn);
  return fn(handle, engineIndex, engineId, engineName, engineNameLen,
            pluginName, pluginNameLen, version, versionLen, type, typeLen);
}

hipdnnStatus_t hipdnnGetVersion_ext(const char** version) {
  using Fn = hipdnnStatus_t (*)(const char**);
  RESOLVE(hipdnnGetVersion_ext, Fn);
  return fn(version);
}

const char* hipdnnVersionString_ext() {
  using Fn = const char* (*)();
  RESOLVE_PTR(hipdnnVersionString_ext, Fn);
  return fn();
}

}  // extern "C"
