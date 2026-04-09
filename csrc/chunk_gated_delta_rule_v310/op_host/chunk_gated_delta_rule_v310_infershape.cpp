/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file chunk_gated_delta_rule_v310_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"

using namespace gert;
namespace ops {

const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t SEQLENS_INDEX = 6;
const size_t STATE_DIM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeChunkGatedDeltaRuleV310(InferShapeContext *context) {
    const Shape *key_shape = context->GetInputShape(KEY_INDEX);
    uint32_t seqLen = key_shape->GetDim(DIM_0);
    uint32_t headDimQK = key_shape->GetDim(DIM_2);
    const Shape *value_shape = context->GetInputShape(VALUE_INDEX);
    uint32_t numHeadV = value_shape->GetDim(DIM_1);
    uint32_t headDimV = value_shape->GetDim(DIM_2);
    const Shape *actual_seq_lengths_shape = context->GetInputShape(SEQLENS_INDEX);
    uint32_t batchSize = actual_seq_lengths_shape->GetDim(DIM_0);
    Shape *core_attn = context->GetOutputShape(0);
    *core_attn = *value_shape;
    auto attrs = context->GetAttrs();
    uint32_t outputFinalState = *(attrs->GetAttrPointer<bool>(1)) ? 1 : 0;
    if (outputFinalState == 1) {
        Shape *last_recurrent_state = context->GetOutputShape(1);
        last_recurrent_state->SetDimNum(STATE_DIM);
        last_recurrent_state->SetDim(DIM_0, batchSize);
        last_recurrent_state->SetDim(DIM_1, numHeadV);
        last_recurrent_state->SetDim(DIM_2, headDimV);
        last_recurrent_state->SetDim(DIM_3, headDimQK);
    }

    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeChunkGatedDeltaRuleV310(InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    auto attrs = context->GetAttrs();
    uint32_t outputFinalState = *(attrs->GetAttrPointer<bool>(1)) ? 1 : 0;
    if (outputFinalState == 1) {
        context->SetOutputDataType(0, ge::DataType::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RecurrentGatedDeltaRuleV310)
    .InferShape(InferShapeRecurrentGatedDeltaRuleV310)
    .InferDataType(InferDataTypeRecurrentGatedDeltaRuleV310);
} // namespace ops
