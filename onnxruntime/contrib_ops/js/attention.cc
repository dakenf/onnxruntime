// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention.h"

namespace onnxruntime {
namespace contrib {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    Attention,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Attention);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
