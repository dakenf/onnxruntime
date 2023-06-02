// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#if defined(_WIN32)
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

bool isLibraryPresent(const std::string& libName)
{
#if defined(_WIN32)
  HMODULE moduleHandle = LoadLibrary(libName.c_str());
  if (moduleHandle != NULL) {
    FreeLibrary(moduleHandle);
    return true;
  }
#elif defined(__unix__) || defined(__APPLE__)
  void* handle = dlopen(libName.c_str(), RTLD_NOW);
  if (handle) {
    dlclose(handle);
    return true;
  }
#endif
  return false;
}

void CheckCudaLibraries (Napi::Env &env)
{
  std::string cudaProvidersLib;
#if defined(_WIN32)
  cudaProvidersLib = "onnxruntime_providers_cuda.dll";
#elif defined(__unix__)
  cudaProvidersLib = "libonnxruntime_providers_cuda.so";
#elif defined(__APPLE__)
  ORT_NAPI_THROW_ERROR(env, "CUDA provider is not supported on macOS");
#endif

  if (!isLibraryPresent(cudaProvidersLib)) {
    ORT_NAPI_THROW_ERROR(env, "CUDA provider lib ", cudaProvidersLib, " could not be found. Please install onnxruntime-gpu release https://github.com/microsoft/onnxruntime/releases");
  }
}

void CheckTensorRtLibraries (Napi::Env &env)
{
  CheckCudaLibraries(env);
  std::string tensorRtProvidersLib;
#if defined(_WIN32)
  tensorRtProvidersLib = "onnxruntime_providers_tensorrt.dll";
#elif defined(__unix__)
  tensorRtProvidersLib = "libonnxruntime_providers_tensorrt.so";
#elif defined(__APPLE__)
  ORT_NAPI_THROW_ERROR(env, "TensorRT provider is not supported on macOS");
#endif

  if (!isLibraryPresent(tensorRtProvidersLib)) {
    ORT_NAPI_THROW_ERROR(env, "TensorRT provider lib ", tensorRtProvidersLib, " could not be found. Please install onnxruntime-gpu release https://github.com/microsoft/onnxruntime/releases");
  }
}
