// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "multi_head_attention.h"

namespace onnxruntime {
namespace contrib {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    MultiHeadAttention,
    kMSDomain,
    1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MultiHeadAttention);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
