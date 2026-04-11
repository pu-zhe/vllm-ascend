#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_ascend._310p.ops import rotary_embedding as rotary_310
from vllm_ascend._310p.ops.rotary_embedding import (
    AscendMRotaryEmbedding310,
    set_mrope_apply_rotary_slices,
)


def _reset_mrope_globals():
    rotary_310._mrope_cos_slice = None
    rotary_310._mrope_sin_slice = None


def _build_mrope_embedding() -> AscendMRotaryEmbedding310:
    emb = AscendMRotaryEmbedding310.__new__(AscendMRotaryEmbedding310)
    emb.mrope_section = [2, 2, 2]
    emb.mrope_interleaved = False
    emb.cos_sin_cache = torch.randn(64, 12, dtype=torch.float32)
    return emb


def test_set_mrope_apply_rotary_slices_populates_globals():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)
    set_mrope_apply_rotary_slices(
        emb.cos_sin_cache,
        positions,
        mrope_section=emb.mrope_section,
        mrope_interleaved=emb.mrope_interleaved,
    )

    assert rotary_310._mrope_cos_slice is not None
    assert rotary_310._mrope_sin_slice is not None
    assert rotary_310._mrope_cos_slice.shape[1] == positions.shape[-1]


def test_set_mrope_apply_rotary_slices_reuses_buffer_address():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)

    set_mrope_apply_rotary_slices(
        emb.cos_sin_cache,
        positions,
        mrope_section=emb.mrope_section,
        mrope_interleaved=emb.mrope_interleaved,
    )
    first_ptr = rotary_310._mrope_cos_slice.data_ptr()

    set_mrope_apply_rotary_slices(
        emb.cos_sin_cache,
        positions,
        mrope_section=emb.mrope_section,
        mrope_interleaved=emb.mrope_interleaved,
    )
    second_ptr = rotary_310._mrope_cos_slice.data_ptr()

    assert first_ptr == second_ptr


def test_apply_rotary_mrope_torch_matches_apply_rotary_emb():
    """Torch MRoPE fallback must match vLLM ApplyRotaryEmb for both neox and GPT-J pairing."""
    t, n_h, rd = 2, 1, 8
    torch.manual_seed(0)
    q_rot = torch.randn(1, t, n_h, rd)
    k_rot = torch.randn(1, t, n_h, rd)
    cos_half = torch.randn(t, rd // 2)
    sin_half = torch.randn(t, rd // 2)
    cos = torch.cat((cos_half, cos_half), dim=-1).view(1, t, 1, rd)
    sin = torch.cat((sin_half, sin_half), dim=-1).view(1, t, 1, rd)

    for neox in (True, False):
        oq, ok = rotary_310._apply_rotary_mrope_torch(
            q_rot.clone(),
            k_rot.clone(),
            cos,
            sin,
            neox,
        )
        rq = ApplyRotaryEmb.forward_static(q_rot[0], cos_half, sin_half, neox)
        rk = ApplyRotaryEmb.forward_static(k_rot[0], cos_half, sin_half, neox)
        assert torch.allclose(oq[0], rq)
        assert torch.allclose(ok[0], rk)


