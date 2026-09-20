"""Loader tests: metadata-only discovery, duplicate detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from tests.fixtures import SLOTS, make_fake_model
from weight_atlas.loaders.safetensors_loader import SafetensorsLoader


def test_loader_discovers_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "m.safetensors"
        make_fake_model(model, n_layers=4, hidden=32)
        loader = SafetensorsLoader()
        handles = loader.open(model)
        # 4 layers * 9 slots + embed + lm_head
        assert len(handles) == 4 * len(SLOTS) + 2
        names = {h.name for h in handles}
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "lm_head.weight" in names


def test_loader_lazy_no_load():
    """Loader must not call get_tensor (i.e., not materialise data)."""
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "m.safetensors"
        make_fake_model(model, n_layers=2, hidden=8)
        loader = SafetensorsLoader()
        handles = loader.open(model)
        for h in handles:
            # shape is known without loading
            assert len(h.shape) == 2
            assert h.shape == (8, 8)


def test_loader_rejects_duplicate_names():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Create two files with the same tensor name.
        from safetensors.numpy import save_file
        a = np.zeros((2, 2), dtype=np.float32)
        save_file({"dup": a}, tmp / "a.safetensors")
        save_file({"dup": a}, tmp / "b.safetensors")
        loader = SafetensorsLoader()
        with pytest.raises(ValueError, match="duplicate tensor name"):
            loader.open(tmp)


def test_loader_skips_amax_sidecars(tmp_path):
    """ModelOpt writes amax.safetensors + amax_checkpoint.safetensors with
    overlapping quantizer keys — they are calibration caches, not weights."""
    w = np.zeros((2, 2), dtype=np.float32)
    q = np.zeros((2,), dtype=np.float32)
    save_file({"model.weight": w}, tmp_path / "model.safetensors")
    save_file({"model.language_model.layers.0.mlp.experts.down_proj.0.input_quantizer": q},
              tmp_path / "amax.safetensors")
    save_file({"model.language_model.layers.0.mlp.experts.down_proj.0.input_quantizer": q},
              tmp_path / "amax_checkpoint.safetensors")
    loader = SafetensorsLoader()
    handles = loader.open(tmp_path)
    assert [h.name for h in handles] == ["model.weight"]


def test_loader_index_selects_weight_shards(tmp_path):
    """With a HF shard index, only weight_map files load — sidecars listed
    nowhere are ignored even when they duplicate a shard tensor name."""
    import json as _json

    w = np.zeros((2, 2), dtype=np.float32)
    save_file({"model.weight": w}, tmp_path / "model-00001-of-00002.safetensors")
    save_file({"other.weight": w}, tmp_path / "model-00002-of-00002.safetensors")
    save_file({"model.weight": w}, tmp_path / "sidecar.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(_json.dumps({
        "weight_map": {
            "model.weight": "model-00001-of-00002.safetensors",
            "other.weight": "model-00002-of-00002.safetensors",
        }
    }))
    loader = SafetensorsLoader()
    handles = loader.open(tmp_path)
    assert sorted(h.name for h in handles) == ["model.weight", "other.weight"]


def test_loader_directory_glob_sorted():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from safetensors.numpy import save_file
        # a.safetensors contains tensor 'y'; z.safetensors contains tensor 'x'.
        # Files are globbed sorted, so a.safetensors is read first → 'y' before 'x'.
        save_file({"x": np.zeros((2, 2), dtype=np.float32)}, tmp / "z.safetensors")
        save_file({"y": np.zeros((2, 2), dtype=np.float32)}, tmp / "a.safetensors")
        loader = SafetensorsLoader()
        handles = loader.open(tmp)
        names = [h.name for h in handles]
        assert names == ["y", "x"]  # sorted by filename (a before z), not tensor name


class TestCorruptHeader:
    def test_oversized_header_rejected(self, tmp_path):
        """A length prefix claiming a multi-GB header must fail loudly, not
        attempt the read."""
        import struct as _struct

        from weight_atlas.loaders.safetensors_loader import _read_header_full

        p = tmp_path / "evil.safetensors"
        p.write_bytes(_struct.pack("<Q", 8 * 1024 * 1024 * 1024) + b"{}")
        with pytest.raises(ValueError, match="corrupt or malicious"):
            _read_header_full(p)

    def test_offsets_outside_data_section_rejected(self, tmp_path):
        """Negative starts would read header bytes as weights; oversized ends
        truncate silently. Both must be rejected up front."""

        from weight_atlas.loaders.safetensors_loader import (
            SafetensorsLoader,
            _read_header_full,
            _validate_offsets,
        )

        tensors = {"model.layers.0.self_attn.q_proj.weight": np.zeros((4, 4), np.float32)}
        p = tmp_path / "m.safetensors"
        save_file(tensors, str(p))

        good_header, off = _read_header_full(p)
        _validate_offsets(p, good_header, off)  # must not raise

        # Negative start.
        bad1 = dict(good_header)
        bad1["model.layers.0.self_attn.q_proj.weight"] = {
            **good_header["model.layers.0.self_attn.q_proj.weight"],
            "data_offsets": [-16, 48],
        }
        with pytest.raises(ValueError, match="invalid data_offsets"):
            _validate_offsets(p, bad1, off)

        # End beyond data section.
        bad2 = dict(good_header)
        bad2["model.layers.0.self_attn.q_proj.weight"] = {
            **good_header["model.layers.0.self_attn.q_proj.weight"],
            "data_offsets": [0, 10 ** 9],
        }
        with pytest.raises(ValueError, match="invalid data_offsets"):
            _validate_offsets(p, bad2, off)

        assert SafetensorsLoader().open(p)  # untouched file still loads
