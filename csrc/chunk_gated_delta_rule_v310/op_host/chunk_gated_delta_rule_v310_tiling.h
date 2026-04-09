/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_v310_tiling.h
 * \brief
 */
#ifndef __OP_HOST_CHUNK_GATED_DELTA_RULE_V310_TILING_H__
#define __OP_HOST_CHUNK_GATED_DELTA_RULE_V310_TILING_H__

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../tiling_base/error_log.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkGatedDeltaRuleV310TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqLen);
  TILING_DATA_FIELD_DEF(uint32_t, numHead);
  TILING_DATA_FIELD_DEF(uint32_t, headDimQK);
  TILING_DATA_FIELD_DEF(uint32_t, headDimV);
  TILING_DATA_FIELD_DEF(uint32_t, chunkSize);
  TILING_DATA_FIELD_DEF(uint32_t, hasInitState);
  TILING_DATA_FIELD_DEF(uint32_t, outputFinalState);
  TILING_DATA_FIELD_DEF(uint32_t, useQKL2normInKernel);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkGatedDeltaRuleV310, ChunkGatedDeltaRuleV310TilingData)

static ge::graphStatus CheckShapeAndUpdate(gert::TilingContext *context, ChunkGatedDeltaRuleV310TilingData &tiling);
static ge::graphStatus CheckTypeAndFormat(gert::TilingContext *context);
static ge::graphStatus CheckAttrAndUpdate(gert::TilingContext *context, ChunkGatedDeltaRuleV310TilingData &tiling);
static ge::graphStatus UpdateMatMulTiling(gert::TilingContext *context, ChunkGatedDeltaRuleV310TilingData &tiling);
static ge::graphStatus TilingFunc(gert::TilingContext *context);
} // namespace optiling
#endif // __OP_HOST_CHUNK_GATED_DELTA_RULE_V310_TILING_H__
