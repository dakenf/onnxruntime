// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common.h"

std::wstring GetBindingPath (Napi::Env env);
bool IsSharedProviderPresent(Napi::Env env, const std::wstring& providerName);
