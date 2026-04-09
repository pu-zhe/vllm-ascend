/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CHUNK_GATED_DELTA_RULE_V310_TORCH_ADPT_H
#define CHUNK_GATED_DELTA_RULE_V310_TORCH_ADPT_H
namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_chunk_gated_delta_rule_310(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& actual_seq_lengths,
    int64_t chunk_size,
    const std::optional<at::Tensor> &initial_state,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel)
{
    auto output_size = op_infer::chunk_gated_delta_rule_npu_output_size(query, value);
    at::Tensor attn = npu_preparation::apply_tensor_with_format(output_size[0], query.options(), ACL_FORMAT_ND);
    at::Tensor state = npu_preparation::apply_tensor_with_format(output_size[1], query.options().dtype(at::kFloat), ACL_FORMAT_ND);
    EXEC_NPU_CMD(aclnnChunkGatedDeltaRule, query, key, value, g, beta, initial_state, actual_seq_lengths, chunk_size, output_final_state, use_qk_l2norm_in_kernel, attn, state);
    return std::tuple<at::Tensor, at::Tensor>(attn, state);
}

}
#endif
