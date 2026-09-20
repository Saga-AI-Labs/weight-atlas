"""Diagnose mapping coverage for a safetensors model dir or a fingerprint.json.

Usage:
    .venv/bin/python scripts/diagnose_mapping.py /path/to/model_dir
    .venv/bin/python scripts/diagnose_mapping.py /path/to/fingerprint.json

Read-only: only safetensors *headers* are read (never payloads), and
fingerprints are only loaded. Nothing is written anywhere.

Reports:
  1. Loader discovery — which files would be scanned (index-aware, amax skip)
  2. Dtype histogram across all headers
  3. Quant-family analysis — NVFP4/HF triples merged vs split across shards,
     ModelOpt quantizer leftovers, stacked-expert patterns
  4. map_name() coverage — slot histogram, top unmapped patterns + samples,
     per-layer dense-slot check (explains main-raster valid_fraction)
"""

from __future__ import annotations

import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from weight_atlas.core.name_map import map_name  # noqa: E402
from weight_atlas.loaders.safetensors_loader import _discover_files  # noqa: E402


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size))


def pattern_of(name: str) -> str:
    """Collapse layer/expert indices so structural families group together."""
    s = re.sub(r"layers?\.\d+", "layers.N", name)
    s = re.sub(r"experts\.\d+", "experts.E", s)
    s = re.sub(r"blk\.\d+", "blk.N", s)
    return s


def report_names(names: list[str], title: str) -> None:
    n = len(names)
    slot_counts: Counter[str] = Counter()
    unmapped: list[str] = []
    for name in names:
        slot = map_name(name)[1]
        slot_counts[slot] += 1
        if slot == "other":
            unmapped.append(name)
    in_slots = n - len(unmapped)
    print(f"\n==== {title}: mapping coverage ====")
    print(f"total: {n}  in_slots: {in_slots} ({in_slots / n:.1%})  other: {len(unmapped)} ({len(unmapped) / n:.1%})")
    print("slot histogram:")
    for slot, count in slot_counts.most_common():
        print(f"  {slot:16s} {count:6d}")
    pats: Counter[str] = Counter(pattern_of(u) for u in unmapped)
    print(f"top unmapped patterns ({len(pats)} distinct):")
    samples: dict[str, list[str]] = {}
    for u in unmapped:
        p = pattern_of(u)
        if len(samples.setdefault(p, [])) < 3:
            samples[p].append(u)
    for pat, count in pats.most_common(15):
        print(f"  {count:6d}  {pat}")
        for s in samples[pat]:
            print(f"             e.g. {s}")
    # Per-layer dense check: which layers lack mapped attention/mlp/router?
    dense_probe = {
        "q_proj": "self_attn.q_proj.weight",
        "o_proj": "self_attn.o_proj.weight",
        "mlp_down": "mlp.down_proj.weight",
        "router": "mlp.gate.weight",
    }
    layers = sorted({map_name(x)[0] for x in names if map_name(x)[0] is not None})
    print(f"layers referenced: {len(layers)}"
          + (f"  range {layers[0]}-{layers[-1]}" if layers else ""))
    for label, suffix in dense_probe.items():
        have = sorted({map_name(x)[0] for x in names
                       if x.endswith(suffix) and map_name(x)[1] != "other"})
        print(f"  layers with mapped {label:8s}: {len(have)}"
              + (f"  e.g. {have[:5]}" if have else "  (NONE)"))


def diagnose_model_dir(model_dir: Path) -> int:
    print(f"model dir: {model_dir}")
    idx_files = sorted(model_dir.glob("*.safetensors.index.json"))
    print(f"index files: {[p.name for p in idx_files] or '(none)'}")
    if idx_files:
        try:
            wm = json.loads(idx_files[0].read_text()).get("weight_map", {})
            print(f"  weight_map entries: {len(wm)}")
        except (OSError, ValueError) as exc:
            print(f"  index unreadable: {exc}")
    all_st = sorted(model_dir.glob("*.safetensors"))
    print(f"all *.safetensors: {len(all_st)}")
    skipped = [p.name for p in all_st if p.name.lower().startswith("amax")]
    if skipped:
        print(f"  amax-skipped ({len(skipped)}): {skipped[:5]}"
              + (" ..." if len(skipped) > 5 else ""))
    try:
        files = _discover_files(model_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"discovery FAILED: {exc}")
        return 1
    print(f"loader would open {len(files)} files: "
          f"{[p.name for p in files[:5]]}{' ...' if len(files) > 5 else ''}")

    headers: dict[str, dict] = {}
    dtype_counts: Counter[str] = Counter()
    per_file_counts: list[tuple[str, int]] = []
    for f in files:
        try:
            h = read_header(f)
        except (OSError, ValueError) as exc:
            print(f"  header read FAILED for {f.name}: {exc}")
            continue
        h.pop("__metadata__", None)
        headers[f.name] = h
        per_file_counts.append((f.name, len(h)))
        for info in h.values():
            dtype_counts[str(info.get("dtype"))] += 1
    total = sum(c for _, c in per_file_counts)
    print(f"\n==== headers: {total} tensors across {len(headers)} files ====")
    print(f"dtype histogram: {dict(dtype_counts)}")
    counts = sorted(c for _, c in per_file_counts)
    if counts:
        print(f"tensors/file: min {counts[0]}  max {counts[-1]}")

    # ---- quant-family analysis ----
    name_to_file: dict[str, str] = {}
    info_of: dict[str, dict] = {}
    for fname, h in headers.items():
        for name, info in h.items():
            name_to_file[name] = fname
            info_of[name] = info
    names = list(name_to_file)

    u8_weights = [x for x in names
                  if x.endswith(".weight")
                  and str(info_of[x].get("dtype")) == "U8"
                  and len(info_of[x].get("shape", [])) == 2]
    print(f"\n==== HF-NVFP4 candidates: {len(u8_weights)} U8 2-D *.weight ====")
    merged = split = 0
    split_examples: list[str] = []
    for w in u8_weights:
        base = w[: -len(".weight")]
        s1, s2 = f"{base}.weight_scale", f"{base}.weight_scale_2"
        same = (s1 in headers[name_to_file[w]] and s2 in headers[name_to_file[w]])
        if same:
            merged += 1
        else:
            split += 1
            if len(split_examples) < 5:
                where = (f"{s1} in {name_to_file.get(s1, '(absent)')}, "
                         f"{s2} in {name_to_file.get(s2, '(absent)')}")
                split_examples.append(f"{w} [{where}]")
    print(f"  triples merged (siblings in same file): {merged}")
    print(f"  triples SPLIT across files/absent: {split}")
    for e in split_examples:
        print(f"    e.g. {e}")
    print(f"  note: the scan loader merges each same-file triple into ONE handle,")
    print(f"  so ~{2 * merged} scale tensors above never reach stats/mapping.")

    for label, pred in [
        ("*.weight_scale (group scales)", lambda x: x.endswith(".weight_scale")),
        ("*.weight_scale_2 (global scales)", lambda x: x.endswith(".weight_scale_2")),
        ("*quantizer* (ModelOpt amax etc.)", lambda x: "quantizer" in x),
        ("*.weight_packed (ct-NVFP4/MXFP4)", lambda x: x.endswith(".weight_packed")),
        ("stacked experts w/o index (experts.PROJ)", lambda x: bool(re.search(r"experts\.(gate|up|down)_proj", x))),
    ]:
        hit = [x for x in names if pred(x)]
        print(f"  {label}: {len(hit)}", end="")
        if hit:
            dtypes = Counter(str(info_of[x].get("dtype")) for x in hit)
            shapes = Counter(tuple(info_of[x].get("shape", ())) for x in hit)
            print(f"  dtypes={dict(dtypes)}")
            print(f"      shapes(top3)={dict(list(shapes.most_common(3)))}")
            print(f"      e.g. {hit[0]}")
        else:
            print()

    report_names(names, "model headers")
    return 0


def diagnose_fingerprint(fp_path: Path) -> int:
    fp = json.loads(fp_path.read_text())
    tensors = fp.get("tensors", {})
    print(f"fingerprint: {fp_path}")
    print(f"tensor records: {len(tensors)}")
    mc = fp.get("mapping_coverage", {})
    if mc:
        print(f"recorded mapping_coverage: in_slots={mc.get('in_slots')} "
              f"unmapped={mc.get('unmapped')}")
    report_names(list(tensors.keys()), "fingerprint tensors")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    target = Path(argv[1])
    if not target.exists():
        print(f"not found: {target}")
        return 2
    if target.is_dir():
        return diagnose_model_dir(target)
    if target.suffix == ".json":
        return diagnose_fingerprint(target)
    print(f"expected a model directory or fingerprint.json, got: {target}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
