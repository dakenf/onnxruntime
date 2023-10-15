// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/operators/conv.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::Conv;

ONNX_OPERATOR_KERNEL_EX(
    NhwcConv,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<true>);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
