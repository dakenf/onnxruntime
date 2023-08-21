// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/bert/attention_base.h"
#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace js {

using onnxruntime::js::JsKernel;
using onnxruntime::contrib::AttentionBase;

class Attention : public JsKernel, AttentionBase {
 public:
  explicit Attention(const OpKernelInfo& info) : JsKernel(info), AttentionBase(info, false) {
    JSEP_INIT_KERNEL_ATTRIBUTE(Attention, ({
                                 "num_heads" : $1,
                                 "is_unidirectional" : $2,
                                 "mask_filter_value" : $3,
                                 "scale" : $4,
                                 "do_rotary" : $5,
                               }),
                               static_cast<int32_t>(num_heads_),
                               static_cast<int32_t>(is_unidirectional_),
                               static_cast<int32_t>(mask_filter_value_),
                               static_cast<int32_t>(scale_),
                               static_cast<int32_t>(do_rotary_));
  }
};

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
