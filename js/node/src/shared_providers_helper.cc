// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"
#if defined(_WIN32)
#include "windows.h"
#endif

std::wstring GetBindingPath (Napi::Env env) {
  DWORD pathLen = MAX_PATH;
  std::wstring path(pathLen, L'\0');
  HMODULE moduleHandle = nullptr;

  GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCSTR>(&GetBindingPath), &moduleHandle);

  DWORD getModuleFileNameResult = GetModuleFileNameW(moduleHandle, const_cast<wchar_t*>(path.c_str()), pathLen);
  while (getModuleFileNameResult == 0 || getModuleFileNameResult == pathLen) {
    DWORD ret = GetLastError();
    if (ret == ERROR_INSUFFICIENT_BUFFER && pathLen < 32768) {
      pathLen *= 2;
      path.resize(pathLen);
      getModuleFileNameResult = GetModuleFileNameW(moduleHandle, const_cast<wchar_t*>(path.c_str()), pathLen);
    } else {
      ORT_NAPI_THROW_ERROR(env, "Failed getting path to load provider libraries, error code: ", ret);
    }
  }

  path.resize(path.rfind(L'\\') + 1);

  return path;
}

bool IsSharedProviderPresent(Napi::Env env, const std::wstring& providerName)
{
#if defined(_WIN32)
  std::wstring libraryName = providerName + L".dll";
#elif defined(__APPLE__)
  std::wstring libraryName = L"lib" + providerName + L".dylib";
#else
  std::wstring libraryName = L"lib" + providerName + L".so";
#endif

#if defined(_WIN32)
  std::wstring fullProviderPath = GetBindingPath(env);
  std::wstring sharedProvidersPath = fullProviderPath;
  sharedProvidersPath.append(L"onnxruntime_providers_shared.dll");

  fullProviderPath.append(libraryName);

  // if we won't load providers_shared first, CUDA or TensorRT libs will return error code 126
  HMODULE sharedModuleHandle = LoadLibraryW(sharedProvidersPath.c_str());
  if (sharedModuleHandle == NULL) {
    return false;
  }

  auto tryLoadLibrary = [](const std::wstring& path) -> bool {
    HMODULE moduleHandle = LoadLibraryW(path.c_str());

    if (moduleHandle != NULL) {
      FreeLibrary(moduleHandle);
      return true;
    }

    DWORD ret = GetLastError();
    // let's treat initialization error as success because library file was found
    return ret == 1114;
  };

  bool loadResult = tryLoadLibrary(libraryName) || tryLoadLibrary(fullProviderPath);
  FreeLibrary(sharedModuleHandle);

  return loadResult;
#else
  void* handle = dlopen(libraryName.c_str(), RTLD_NOW);
  if (handle != NULL) {
    dlclose(handle);
    return true;
  }
  return false;
#endif
}
