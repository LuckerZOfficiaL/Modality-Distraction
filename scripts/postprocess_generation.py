"""Post-process oracle generation outputs: deterministically shuffle MCQ options.

Reads every chunk_*.result.jsonl under data/dm/work/gen/ and writes a single
data/dm/candidates.jsonl with:
  - options permuted per (seed_id, label) using a seeded RNG
  - correct_index updated to match the new position
  - both vision and text candidates exploded into separate rows
  - candidate_id = f"{seed_id}_v" or f"{seed_id}_t"

Row schema of candidates.jsonl:
  {
    "candidate_id": "<seed_id>_v" | "<seed_id>_t",
    "seed_id": "...",
    "image_path": "...",                 # filled from seeds.jsonl
    "original_caption": "...",           # filled from seeds.jsonl
    "label": "vision" | "text",
    "question": "...",
    "options": [...],                    # shuffled
    "correct_index": <0|1|2|3>,          # post-shuffle
    "caption_for_filter": "...",         # original for vision, augmented for text
    "rationale": "..."
  }
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import yaml


FORBIDDEN_RE_DEFAULT = re.compile(
    r"\b(image|picture|photo|photograph|caption|text|description|shown|"
    r"depicted|visible|pictured|displayed|according to|based on)\b",
    re.IGNORECASE,
)

# VisText: chart-vocab tells must also be banned, since they would let a reader
# distinguish vision-grounded from text-grounded questions without seeing
# either modality.
FORBIDDEN_RE_VISTEXT = re.compile(
    r"\b(image|picture|photo|photograph|caption|text|description|shown|"
    r"depicted|visible|pictured|displayed|illustrated|according to|based on|"
    r"chart|plot|graph|figure|diagram|visualization|"
    r"bar|line|axis|y-axis|x-axis|tick|legend|slice|wedge|series)\b",
    re.IGNORECASE,
)


def _shuffle(options: list[str], correct_index: int, rng: random.Random) -> tuple[list[str], int]:
    order = list(range(len(options)))
    rng.shuffle(order)
    new_options = [options[i] for i in order]
    new_correct = order.index(correct_index)
    return new_options, new_correct


def _norm(s: str) -> str:
    """Lowercase + collapse whitespace; for case-insensitive substring checks."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def verify_text_block(block: dict) -> str | None:
    """Apply C1/C2 + two-stage bridge checks. Return reason if invalid, else None.

    Oracle-agnostic: operates purely on the JSON the oracle produced.
    Backward-compatible: rows missing `bridge_entity` / `derived_fact` skip the
    bridge structural checks but STILL get the C1 substring safety net on the
    correct option.
    """
    aug = _norm(block.get("augmented_caption", ""))
    opts = block.get("options") or []
    ci = block.get("correct_index")
    if not opts or ci is None or not (0 <= int(ci) < len(opts)):
        return "bad-options-or-correct_index"
    correct = _norm(str(opts[int(ci)]))
    if not correct:
        return "empty-correct-option"

    # C1 safety net (applies regardless of schema version):
    # the correct-option string must not be a contiguous substring of augmented_caption.
    if correct and correct in aug:
        return "C1-answer-substring-of-augmented_caption"

    # New two-stage schema: also enforce bridge_entity/derived_fact contract.
    if "derived_fact" in block or "bridge_entity" in block:
        derived = _norm(str(block.get("derived_fact", "")))
        bridge = _norm(str(block.get("bridge_entity", "")))
        if not derived:
            return "missing-derived_fact"
        if not bridge:
            return "missing-bridge_entity"
        if derived != correct:
            return "derived_fact-mismatch-correct-option"
        if derived in aug:
            return "C1-derived_fact-substring-of-augmented_caption"
        if bridge not in aug:
            return "bridge_entity-not-in-augmented_caption"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    seed = int(cfg["seed"])

    # index seeds for image_path / caption lookup
    seeds = {}
    sources = set()
    for l in (dm_dir / "seeds.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            seeds[r["seed_id"]] = r
            sources.add(r.get("source", "dci"))

    forbidden_re = FORBIDDEN_RE_VISTEXT if sources == {"vistext"} else FORBIDDEN_RE_DEFAULT
    print(f"sources={sources}; using {'VISTEXT' if forbidden_re is FORBIDDEN_RE_VISTEXT else 'DEFAULT'} forbidden-word regex")

    work_dir = dm_dir / "work" / "gen"
    chunks = sorted(work_dir.glob("chunk_*.result.jsonl"))
    per_seed_dir = work_dir / "seeds"
    per_seed_files = sorted(per_seed_dir.glob("*.out.json")) if per_seed_dir.is_dir() else []
    candidates_path = dm_dir / "candidates.jsonl"

    # Yield raw oracle rows from both legacy chunk JSONL and new per-seed JSON files.
    # Per-seed files take precedence (oracle-agnostic; either backend may produce them).
    seen_seed_ids: set[str] = set()

    def iter_rows():
        for p in per_seed_files:
            try:
                row = json.loads(p.read_text())
            except Exception:
                yield {"error": f"unparseable per-seed file: {p.name}"}
                continue
            if "seed_id" in row:
                seen_seed_ids.add(row["seed_id"])
            yield row
        for ch in chunks:
            for line in ch.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    yield {"error": "unparseable chunk line"}
                    continue
                if row.get("seed_id") in seen_seed_ids:
                    continue  # per-seed file wins
                yield row

    n_in = n_err = n_out = n_forbidden = n_bridge_drop = 0
    bridge_drop_reasons: dict[str, int] = {}
    with candidates_path.open("w") as out:
        for row in iter_rows():
            n_in += 1
            if "error" in row:
                n_err += 1
                continue
            # newer chunks wrap MCQ in a "result" key; flatten for uniform handling
            if "result" in row and isinstance(row["result"], dict):
                inner = row["result"]
                flat = {"seed_id": row["seed_id"]}
                if "vision" in inner: flat["vision"] = inner["vision"]
                if "text"   in inner: flat["text"]   = inner["text"]
                row = flat
            # T-only mode produces rows without a "vision" key; that's fine.
            if "vision" not in row and "text" not in row:
                n_err += 1
                continue
            sid = row.get("seed_id")
            seed_row = seeds.get(sid)
            if seed_row is None:
                n_err += 1
                continue

            # drop the row if any present question contains a forbidden word
            vq = row.get("vision", {}).get("question", "") if "vision" in row else ""
            tq = row.get("text",   {}).get("question", "") if "text"   in row else ""
            if forbidden_re.search(vq) or forbidden_re.search(tq):
                n_forbidden += 1
                continue

            # C1/C2 + bridge structural check (T-side only; oracle-agnostic).
            if "text" in row:
                reason = verify_text_block(row["text"])
                if reason is not None:
                    n_bridge_drop += 1
                    bridge_drop_reasons[reason] = bridge_drop_reasons.get(reason, 0) + 1
                    # Strip the bad text block; keep vision if present.
                    row = {k: v for k, v in row.items() if k != "text"}
                    if "vision" not in row:
                        continue

            labels_to_emit = []
            if "vision" in row:
                labels_to_emit.append(("vision", row["vision"], seed_row["caption"]))
            if "text" in row:
                labels_to_emit.append(("text", row["text"],
                                       row["text"].get("augmented_caption", seed_row["caption"])))
            for label, block, cap_for_filter in labels_to_emit:
                rng = random.Random(f"{seed}|{sid}|{label}")
                opts, ci = _shuffle(block["options"], int(block["correct_index"]), rng)
                out.write(json.dumps({
                    "candidate_id": f"{sid}_{label[0]}",
                    "seed_id": sid,
                    "image_path": seed_row["image_path"],
                    "original_caption": seed_row["caption"],
                    "label": label,
                    "question": block["question"],
                    "options": opts,
                    "correct_index": ci,
                    "caption_for_filter": cap_for_filter,
                    "rationale": block.get("rationale", ""),
                }) + "\n")
                n_out += 1

    # report distribution
    import collections
    dist = collections.Counter()
    for l in candidates_path.read_text().splitlines():
        dist[json.loads(l)["correct_index"]] += 1
    print(f"read {n_in} rows from {len(per_seed_files)} per-seed files + {len(chunks)} chunk files "
          f"({n_err} errors, {n_forbidden} dropped for forbidden words, "
          f"{n_bridge_drop} T-blocks dropped by C1/bridge verifier). "
          f"wrote {n_out} candidates -> {candidates_path}")
    if bridge_drop_reasons:
        print(f"  bridge_drop reasons: {bridge_drop_reasons}")
    print(f"correct_index distribution: {dict(dist)}")


if __name__ == "__main__":
    main()
