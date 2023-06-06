// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_DML) && defined(_WIN32)
#include "shared_providers_helper.h"
#include "windows.h"

void LoadDirectMLDll(Napi::Env env)
{
  std::wstring path = GetBindingPath(env);
  path.resize(path.rfind(L'\\') + 1);
  path.append(L"DirectML.dll");
  HMODULE libraryLoadResult = LoadLibraryW(path.c_str());

  if (libraryLoadResult == NULL) {
    int const ret = GetLastError();
    ORT_NAPI_THROW_ERROR(env, "Failed loading bundled DirectML.dll, error code: ", ret);
  }
}
#endif
