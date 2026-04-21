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
    const at::Tensor& actual_seq_lengths)
{
    at::Tensor out = at::empty(value.sizes(), value.options());
    // Prefer CPU tensor from caller to avoid device-side overhead, but kernel
    // needs device-accessible memory. Copy to NPU if needed.
    at::Tensor actual_seq_lengths_dev = actual_seq_lengths;
    if (actual_seq_lengths_dev.device().is_cpu()) {
        actual_seq_lengths_dev = actual_seq_lengths_dev.to(query.device(), /*non_blocking=*/false);
    }
    if (!actual_seq_lengths_dev.is_contiguous()) {
        actual_seq_lengths_dev = actual_seq_lengths_dev.contiguous();
    }
    TORCH_CHECK(actual_seq_lengths_dev.scalar_type() == at::kInt,
                "actual_seq_lengths must be int32");

    auto batch_size = actual_seq_lengths_dev.sizes()[0];
    auto headnum = value.sizes()[1];
    auto headdim_k = key.sizes()[2];
    auto headdim_v = value.sizes()[2];
    at::Tensor final_state = at::empty({batch_size, headnum, headdim_k, headdim_v}, value.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnChunkGatedDeltaRuleV310, query, key, value, g, beta, actual_seq_lengths_dev, out, final_state);
    return {out, final_state};
}

}
#endif
