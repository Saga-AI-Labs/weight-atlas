"""Tests for mapping_coverage format in fingerprint.json."""

from __future__ import annotations

from pathlib import Path

import pytest

from weight_atlas.scan import _build_fingerprint


@pytest.fixture
def spec():
    from weight_atlas.core.types import AtlasSpec
    return AtlasSpec.from_json(Path("specs/atlas_spec.v2.2.json"))


def test_mapping_coverage_format(spec):
    """mapping_coverage must have in_slots (ratio), in_other (ratio), unmapped (count), unmapped_tensors (list)."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name="blk.0.attn_q.weight", shape=(4096, 4096)),
        TensorStats(name="blk.0.attn_k.weight", shape=(4096, 4096)),
        TensorStats(name="blk.0.ffn_gate.weight", shape=(4096, 12288)),
        TensorStats(name="unknown_tensor.weight", shape=(100,)),
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert "in_slots" in mc
    assert "in_other" in mc
    assert "unmapped" in mc
    assert "unmapped_tensors" in mc

    # in_slots and in_other should be ratios (floats between 0 and 1)
    assert 0.0 <= mc["in_slots"] <= 1.0
    assert 0.0 <= mc["in_other"] <= 1.0

    # unmapped should be a count
    assert isinstance(mc["unmapped"], int)
    assert mc["unmapped"] == 1

    # unmapped_tensors should be a list
    assert isinstance(mc["unmapped_tensors"], list)
    assert "unknown_tensor.weight" in mc["unmapped_tensors"]


def test_mapping_coverage_all_mapped(spec):
    """When all tensors are mapped, in_slots should be 1.0."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name="blk.0.attn_q.weight", shape=(4096, 4096)),
        TensorStats(name="blk.0.attn_k.weight", shape=(4096, 4096)),
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0
    assert mc["in_other"] == 0.0
    assert mc["unmapped"] == 0
    assert mc["unmapped_tensors"] == []


def test_mapping_coverage_none_mapped(spec):
    """When no tensors are mapped, in_slots should be 0.0."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name="unknown_tensor_1.weight", shape=(100,)),
        TensorStats(name="unknown_tensor_2.weight", shape=(200,)),
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 0.0
    assert mc["in_other"] == 1.0
    assert mc["unmapped"] == 2
    assert len(mc["unmapped_tensors"]) == 2


def test_mapping_coverage_mtp_draft_head_mapped(spec):
    """Qwen3-Next MTP draft head tensors (blk.N.nextn.*) must map to mtp slots."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name="blk.64.nextn.eh_proj.weight", shape=(10240, 5120)),
        TensorStats(name="blk.64.nextn.enorm.weight", shape=(5120,)),
        TensorStats(name="blk.64.nextn.hnorm.weight", shape=(5120,)),
        TensorStats(name="blk.64.nextn.shared_head_norm.weight", shape=(5120,)),
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0
    assert mc["in_other"] == 0.0
    assert mc["unmapped"] == 0
    assert mc["unmapped_tensors"] == []


def test_mapping_coverage_attn_sinks_mapped(spec):
    """GPT-OSS / Qwen3-Next attention-sink registers (blk.N.attn_sinks) must map."""
    from weight_atlas.core.types import TensorStats

    stats = [
        TensorStats(name=f"blk.{i}.attn_sinks.weight", shape=(64,))
        for i in range(36)
    ]
    fp = _build_fingerprint(stats, spec, "gguf")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0
    assert mc["in_other"] == 0.0
    assert mc["unmapped"] == 0
    assert mc["unmapped_tensors"] == []


def test_mapping_coverage_flash_next_hf_families(spec):
    """Qwen3.8-Flash-Next HF export (Unsloth plefp8): PLE projections,
    indexer norms, hyper-connection mixers and MTP tower must all map —
    and MTP must stay off the language raster (layer None)."""
    from weight_atlas.core.types import TensorStats

    names = [
        "model.language_model.layers.1.ple.key_proj.weight",
        "model.language_model.layers.1.ple.value_proj.weight",
        "model.language_model.layers.1.ple.conv1d.weight",
        "model.language_model.layers.1.ple.norm_conv.weight",
        "model.language_model.layers.1.ple.norm_key.weight",
        "model.language_model.layers.1.ple.norm_query.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_30.weight",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
        "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight",
        "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight",
        "model.language_model.layers.3.attn_hyper_connection.block_inject_weight.weight",
        "model.language_model.layers.3.attn_hyper_connection.hc_norm.weight",
        "model.language_model.layers.3.mlp_hyper_connection.input_mix_weight_up.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    ]
    stats = [TensorStats(name=n, shape=(32, 32)) for n in names]
    fp = _build_fingerprint(stats, spec, "safetensors")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0, mc["unmapped_tensors"]
    assert mc["unmapped"] == 0

    from weight_atlas.core.name_map import map_name

    # PLE table material is non-layer (like the GGUF giant table)
    layer, slot = map_name(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_30.weight"
    )
    assert (layer, slot) == (None, "ngram_embd")
    # ...while per-layer PLE projections keep their layer cells
    assert map_name("model.language_model.layers.1.ple.key_proj.weight") == (1, "ngram_key")


def test_mapping_coverage_nvfp4_scale_siblings_mapped(spec):
    """NVFP4/HF scale siblings that survive as standalone handles (split
    triples, per-expert input scales) must map to the quant_scale slot —
    not "other" (Flash-Next ablits carry ~73k of them)."""
    from weight_atlas.core.types import TensorStats

    names = [
        "model.language_model.layers.0.mlp.experts.0.down_proj.input_scale",
        "model.language_model.layers.1.mlp.experts.261.down_proj.weight_scale_2",
        "model.language_model.layers.5.mlp.experts.309.gate_proj.weight_scale",
        "model.language_model.layers.5.mlp.experts.309.gate_proj.weight_scale_2",
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight_global_scale",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
    ]
    stats = [TensorStats(name=n, shape=(32, 32)) for n in names]
    fp = _build_fingerprint(stats, spec, "safetensors")

    mc = fp["mapping_coverage"]
    assert mc["in_slots"] == 1.0, mc["unmapped_tensors"]
    assert mc["unmapped"] == 0

    from weight_atlas.core.name_map import map_name

    assert map_name(names[0]) == (0, "quant_scale")
    assert map_name(names[1]) == (1, "quant_scale")
    assert map_name(names[2]) == (5, "quant_scale")
    assert map_name(names[5]) == (0, "shared_gate")


def test_mapping_canonical_spec_agrees_on_aux_slots():
    """The canonical v2.4 spec block (default-spec path) must resolve the
    aux slots identically to the fallback tables."""
    from weight_atlas.core.name_map import map_name

    assert map_name(
        "model.language_model.layers.1.mlp.experts.261.down_proj.weight_scale"
    ) == (1, "quant_scale")
    assert map_name(
        "model.language_model.layers.0.mlp.shared_expert_gate.weight"
    ) == (0, "shared_gate")
    # MTP tower stays off the language raster (layer None) with the aux slot.
    assert map_name(
        "mtp.layers.0.mlp.experts.0.down_proj.weight_scale"
    ) == (None, "quant_scale")
    # Merged expert weights still resolve the expert machinery.
    from weight_atlas.core.name_map import (
        extract_expert_id,
        get_moe_slot,
        is_expert_tensor,
    )

    w = "model.language_model.layers.0.mlp.experts.7.up_proj.weight"
    assert map_name(w) == (0, "expert")
    assert is_expert_tensor(w) and get_moe_slot(w) == "up"
    assert extract_expert_id(w) == 7


def test_mapping_hf_moe_rules_in_sync_with_canonical_spec():
    """The fallback hf.moe table must equal the canonical v2.4 block (the
    group this feature edits). Full-table equality is NOT asserted: the GGUF
    groups have pre-existing drift (out of scope)."""
    import json

    from weight_atlas.core.name_map import _MOE_RULES

    block = json.loads(Path("specs/atlas_spec.v2.4.json").read_text())["name_map"]
    assert block["conventions"]["hf"]["rules"]["moe"] == [
        [p.pattern, s] for p, s in _MOE_RULES
    ]


def test_quant_scale_and_shared_gate_excluded_from_fields(spec):
    """Aux slots add no raster columns and never enter expert panels: the
    main grid keeps exactly len(spec.slots) columns."""
    import numpy as np

    from weight_atlas.core.name_map import (
        extract_expert_id,
        get_moe_slot,
        is_expert_tensor,
    )
    from weight_atlas.core.types import TensorStats
    from weight_atlas.fields.rasterizer import (
        rasterize,
        rasterize_expert_panels,
    )

    stats = [
        TensorStats(name="model.layers.0.self_attn.q_proj.weight", shape=(4, 4), frobenius=1.0),
        TensorStats(name="model.layers.0.mlp.experts.0.down_proj.input_scale", shape=(4, 4), frobenius=2.0),
        TensorStats(name="model.layers.0.mlp.experts.0.down_proj.weight_scale", shape=(4, 4), frobenius=3.0),
        TensorStats(name="model.layers.0.mlp.shared_expert_gate.weight", shape=(4, 4), frobenius=4.0),
        TensorStats(name="model.layers.0.mlp.experts.0.down_proj.weight", shape=(4, 4), frobenius=5.0),
    ]
    field = rasterize(stats, spec, "frobenius")
    assert field.data.shape == (1, len(spec.slots))
    assert list(field.col_labels) == list(spec.slots)
    # Only the dense q_proj cell is filled; aux tensors fill nothing.
    assert float(np.nansum(field.data)) == 1.0

    panels = rasterize_expert_panels(stats, spec, "frobenius")
    assert panels  # the real expert weight still builds panels
    total = sum(float(np.nansum(p.data)) for p in panels)
    assert total == 5.0  # scales contribute nothing

    for n in (
        "model.layers.0.mlp.experts.0.down_proj.input_scale",
        "model.layers.0.mlp.experts.0.down_proj.weight_scale",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ):
        assert not is_expert_tensor(n)
        assert get_moe_slot(n) is None
        assert extract_expert_id(n) is None


def test_mapping_mtp_never_collides_with_language_rows(spec):
    """MTP draft-tower tensors must map layer-None: mtp.layers.N would
    otherwise overwrite language row N (raster cells are last-wins)."""
    from weight_atlas.core.name_map import map_name

    for n in (
        "mtp.layers.3.self_attn.q_proj.weight",
        "mtp.layers.3.mlp.shared_expert.down_proj.weight",
        "mtp.layers.3.attn_hyper_connection.hc_norm.weight",
        "mtp.fc_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ):
        layer, slot = map_name(n)
        assert layer is None, f"{n} got language row {layer}"
        assert slot != "other" or "fc_embedding" in n or "pre_fc_norm" in n, n
