/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_v310_tiling.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_v310_tiling.h"

namespace optiling {
static ge::graphStatus CheckShapeAndUpdate(gert::TilingContext *context, ChunkGatedDeltaRuleV310TilingData &tiling) {
    if (context == nullptr) {
        OP_LOGE(context_->GetNodeName() , "context is null\n");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *query_shape = context->GetInputShape(0);
    if (query_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "query is null\n");
        return ge::GRAPH_FAILED;
    }
    if (query_shape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(context_->GetNodeName() , "query dim is not 3\n");
        return ge::GRAPH_FAILED;
    }
    uint32_t seqLen = query_shape->GetStorageShape().GetDim(0);
    uint32_t numHead = query_shape->GetStorageShape().GetDim(1);
    uint32_t headDimQK = query_shape->GetStorageShape().GetDim(2);
    OP_LOGE(context_->GetNodeName() , "query shape is (%d, %d, %d)\n", seqLen, numHead, headDimQK);

    if (numHead % 16 != 0) {
        OP_LOGE(context_->GetNodeName() , "Number of head must be a multiple of 16.\n");
        return ge::GRAPH_FAILED;
    }

    if ((seqLen * numHead * 4) > (64*64*6*4)) {
        OP_LOGE(context_->GetNodeName() , "The product of seqLen and numHead must be less than %d bytes.\n", 64*64*6*4);
        return ge::GRAPH_FAILED;
    }

    if (headDimQK != 128){
        OP_LOGE(context_->GetNodeName() , "The head dimension of query and key must be 128. \n");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *key_shape = context->GetInputShape(1);
    if (key_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "key is null\n");
        return ge::GRAPH_FAILED;
    }
    if (key_shape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(context_->GetNodeName() , "key dim is not 3\n");
        return ge::GRAPH_FAILED;
    }
    if (seqLen != key_shape->GetStorageShape().GetDim(0) || numHead != key_shape->GetStorageShape().GetDim(1) || headDimQK != key_shape->GetStorageShape().GetDim(2)) {
        OP_LOGE(context_->GetNodeName() , "key shape is illegal\n");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *value_shape = context->GetInputShape(2);
    if (value_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "value is null\n");
        return ge::GRAPH_FAILED;
    }
    if (value_shape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(context_->GetNodeName() , "value dim is not 3\n");
        return ge::GRAPH_FAILED;
    }
    if (seqLen != value_shape->GetStorageShape().GetDim(0)) {
        OP_LOGE(context_->GetNodeName() , "value shape is illegal\n");
        return ge::GRAPH_FAILED;
    }
    uint32_t numHeadV = value_shape->GetStorageShape().GetDim(1);
    uint32_t headDimV = value_shape->GetStorageShape().GetDim(2);
    if if (numHeadV != numHead) {
        OP_LOGE(context_->GetNodeName() , "numHeadV %d should be same with numHead %d\n", numHeadV, numHead);
        return ge::GRAPH_FAILED;
    }
    OP_LOGE(context_->GetNodeName() , "value shape is (%d, %d, %d)\n", seqLen, numHeadV, headDimV);

    const gert::StorageShape *g_shape = context->GetInputShape(3);
    if (g_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "g is null\n");
        return ge::GRAPH_FAILED;
    }
    if (g_shape->GetStorageShape().GetDimNum() != 2) {
        OP_LOGE(context_->GetNodeName() , "g dim is not 2\n");
        return ge::GRAPH_FAILED;
    }
    if (seqLen != g_shape->GetStorageShape().GetDim(0) || numHeadV != g_shape->GetStorageShape().GetDim(1)) {
        OP_LOGE(context_->GetNodeName() , "g shape is illegal\n");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *beta_shape = context->GetInputShape(4);
    if (beta_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "beta is null\n");
        return ge::GRAPH_FAILED;
    }
    if (beta_shape->GetStorageShape().GetDimNum() != 2) {
        OP_LOGE(context_->GetNodeName() , "beta dim is not 3\n");
        return ge::GRAPH_FAILED;
    }
    if (seqLen != beta_shape->GetStorageShape().GetDim(0) || numHeadV != beta_shape->GetStorageShape().GetDim(1)) {
        OP_LOGE(context_->GetNodeName() , "beta shape is illegal\n");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *actual_seq_lengths_shape = context->GetInputShape(6);
    if (actual_seq_lengths_shape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE(context_->GetNodeName() , "actual seq lengths dim is not 1\n");
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = actual_seq_lengths_shape->GetStorageShape().GetDim(0);

    const gert::StorageShape *init_state_shape = context->GetInputShape(5);
    uint32_t hasInitState = 1;
    if (init_state_shape == nullptr) {
        hasInitState = 0;
    }
    if (hasInitState) {
        if (init_state_shape->GetStorageShape().GetDimNum() != 4) {
            OP_LOGE(context_->GetNodeName() , "init state dim is not 4\n");
            return ge::GRAPH_FAILED;
        }
        if (batchSize != init_state_shape->GetStorageShape().GetDim(0) || numHeadV != init_state_shape->GetStorageShape().GetDim(1) || headDimV != init_state_shape->GetStorageShape().GetDim(2) || headDimQK != init_state_shape->GetStorageShape().GetDim(3)) {
            OP_LOGE(context_->GetNodeName() , "init state shape is illegal\n");
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGE(context_->GetNodeName() , "hasInitState is %d\n", hasInitState);

    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_numHead(numHead);
    tiling.set_headDimQK(headDimQK);
    tiling.set_numheadV(numHeadV);
    tiling.set_headDimV(headDimV);
    tiling.set_hasInitState(hasInitState);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTypeAndFormat(gert::TilingContext *context) {
    const gert::Tensor *query_tensor = context->GetInputTensor(0);
    if (query_tensor->GetDataType() != ge::DataType::DT_FLOAT16) {
        OP_LOGE(context_->GetNodeName() , "query is not fp16\n");
        return ge::GRAPH_FAILED;
    }
    if (query_tensor->GetStorageFormat() != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName() , "query is not ND format\n");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *key_tensor = context->GetInputTensor(1);
    if (key_tensor->GetDataType() != ge::DataType::DT_FLOAT16) {
        OP_LOGE(context_->GetNodeName() , "key is not fp16\n");
        return ge::GRAPH_FAILED;
    }
    if (key_tensor->GetStorageFormat() != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName() , "key is not ND format\n");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *value_tensor = context->GetInputTensor(2);
    if (value_tensor->GetDataType() != ge::DataType::DT_FLOAT16) {
        OP_LOGE(context_->GetNodeName() , "value is not fp16\n");
        return ge::GRAPH_FAILED;
    }
    if (value_tensor->GetStorageFormat() != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName() , "value is not ND format\n");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *g_tensor = context->GetInputTensor(3);
    if (g_tensor->GetDataType() != ge::DataType::DT_FLOAT) {
        OP_LOGE(context_->GetNodeName() , "g is not fp16\n");
        return ge::GRAPH_FAILED;
    }
    if (g_tensor->GetStorageFormat() != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName() , "g is not ND format\n");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *beta_tensor = context->GetInputTensor(4);
    if (beta_tensor->GetDataType() != ge::DataType::DT_FLOAT16) {
        OP_LOGE(context_->GetNodeName() , "beta is not fp16\n");
        return ge::GRAPH_FAILED;
    }
    if (beta_tensor->GetStorageFormat() != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName() , "beta is not ND format\n");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *init_state_tensor = context->GetInputTensor(5);
    if (init_state_tensor != nullptr) {
        if (init_state_tensor->GetDataType() != ge::DataType::DT_FLOAT) {
            OP_LOGE(context_->GetNodeName() , "init state is not fp32\n");
            return ge::GRAPH_FAILED;
        }
        if (init_state_tensor->GetStorageFormat() != ge::Format::FORMAT_ND) {
            OP_LOGE(context_->GetNodeName() , "init state is not ND format\n");
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Tensor *actual_seq_lengths_tensor = context->GetInputTensor(6);
    if (actual_seq_lengths_tensor->GetDataType() != ge::DataType::DT_INT32) {
        OP_LOGE(context_->GetNodeName() , "actual seq lengths is not int32\n");
        return ge::GRAPH_FAILED;
    }
    if (actual_seq_lengths_tensor->GetStorageFormat() != ge::Format::FORMAT_ND) {
        OP_LOGE(context_->GetNodeName() , "actual seq lengths is not ND format\n");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrAndUpdate(gert::TilingContext *context, ChunkGatedDeltaRuleV310TilingData &tiling) {
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE("attrs is null\n", context->GetNodeName());
        return ge::GRAPH_FAILED;
    }
    if (attrs->GetAttrNum() != 3) {
        OP_LOGE("attrs num is not 3\n", context->GetNodeName());
        return ge::GRAPH_FAILED;
    }
    uint32_t chunkSize = *(attrs->GetAttrPointer<uint32_t>(0));
    if (chunkSize != 64 && chunkSize != 1) {
        OP_LOGE("only support chunkSize = 64 and 1\n", context->GetNodeName());
        return ge::GRAPH_FAILED;
    }

    uint32_t outputFinalState = *(attrs->GetAttrPointer<bool>(1)) ? 1 : 0;
    uint32_t useQKL2normInKernel = *(attrs->GetAttrPointer<bool>(2)) ? 1 : 0;

    OP_LOGE(context_->GetNodeName() , "chunkSize is %d, outputFinalState is %d, useQKL2normInKernel is %d\n", chunkSize, outputFinalState, useQKL2normInKernel);
    tiling.set_chunkSize(chunkSize);
    tiling.set_outputFinalState(outputFinalState);
    tiling.set_useQKL2normInKernel(useQKL2normInKernel);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus UpdateMatMulTiling(gert::TilingContext *context, ChunkGatedDeltaRuleV310TilingData &tiling) {
    using namespace matmul_tiling;
    int M = 64;
    int N = 128;
    int K = 64;
    matmul_tiling::TPosition leftPosition = matmul_tiling::TPosition::VECIN;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    bool isTransA = false;

    TPosition rightPosition = TPosition::VECIN;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    bool isTransB = false;

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;

    TPosition biasPosition = TPosition::VECOUT;
    CubeFormat biasFormat = CubeFormat::ND;
    DataType biasDtype = DataType::DT_FLOAT;
    bool isBias = false;

    int usedCoreNum = 1;
    int baseM = 64;
    int baseN = 128;

    TCubeTiling &tilingData = tiling.matmulTiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MultiCoreMatmulTiling tilingApi(ascendcPlatform);

    tilingApi.SetDim(usedCoreNum); // Set the number of cores that participate in multi-core computaion is 2.
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);
    tilingApi.SetBiasType(biasPosition, biasFormat, biasDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(isBias);
    tilingApi.SetTraverse(MatrixTraverse::FIRSTM); // Set the matmul travse is FIRSTM.
    tilingApi.SetFixSplit(baseM, baseN, -1);       // Set the fixed baseM=128, baseN=256.
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData); // Get matmul tiling data.
    tilingData.set_stepM(1);                       // Set the matmul tiling stepM=1.
    tilingData.set_stepN(1);                       // Set the matmul tiling stepN=1.
    if (res == -1) {
        OP_LOGE("gen matmul tiling failed\n", context->GetNodeName());
        return ge::GRAPH_FAILED;
    }
    SysTilingTempBufSize bufSize;
    MatmulGetTmpBufSize(tilingData, bufSize);
    ASC_CPU_LOG_INFO("ubSize=%d\n",bufSize.ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    ChunkGatedDeltaRuleV310TilingData tiling;

    if (CheckShapeAndUpdate(context, tiling) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckTypeAndFormat(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttrAndUpdate(context, tiling) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(8);

    if (UpdateMatMulTiling(context, tiling) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t usrSize = 128 * 128 * 4 * 10;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
