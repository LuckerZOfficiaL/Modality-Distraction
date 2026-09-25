"""Build qwen-pass subsets from 04c predictions on multidomain_v1.

train_pass = items in splits/train where qwen got V+T correct.
val_pass   = items in splits/val   where qwen got V+T correct.
test       = full splits/test (no filtering — eval uses test as-is).

Also writes per-source / per-label / per-split keep-rate stats.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DM_DIR = ROOT / "data/dm/multidomain_v1"
PRED_PATH = DM_DIR / "pass_qwen_candidates" / "predictions.jsonl"
PASS_DIR = DM_DIR / "pass"


def main() -> None:
    PASS_DIR.mkdir(parents=True, exist_ok=True)

    preds = {}
    for line in PRED_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["candidate_id"]] = r

    stats = {}
    for split in ("train", "val"):
        rows = [json.loads(l)
                for l in (DM_DIR / "splits" / f"{split}.jsonl").read_text().splitlines() if l.strip()]
        kept = []
        per_src_total = Counter()
        per_src_kept = Counter()
        for r in rows:
            cid = r["candidate_id"]
            src = r.get("source", "?")
            label = r["label"]
            per_src_total[(src, label)] += 1
            p = preds.get(cid)
            if p is None:
                continue
            ci = p["correct_index"]
            vt_ok = p["vt"] == ci
            v_ok = p["v"] == ci
            t_ok = p["t"] == ci
            # strict 3-pass behavioral filter: modality isolation
            if label == "vision":
                keep = vt_ok and v_ok and not t_ok
            else:  # text
                keep = vt_ok and t_ok and not v_ok
            if keep:
                kept.append(r)
                per_src_kept[(src, label)] += 1
        out_path = PASS_DIR / f"{split}.jsonl"
        with out_path.open("w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        stats[f"{split}_pass"] = {
            "n_total": len(rows),
            "n_kept": len(kept),
            "keep_rate": len(kept) / max(len(rows), 1),
            "per_source": {f"{s}/{l}": {"total": per_src_total[(s, l)],
                                        "kept": per_src_kept[(s, l)]}
                           for (s, l) in per_src_total},
        }
        print(f"{split}_pass: {len(kept)}/{len(rows)}  ({100*len(kept)/max(len(rows),1):.1f}%)")
        for (s, l), tot in sorted(per_src_total.items()):
            kp = per_src_kept[(s, l)]
            print(f"  {s:8s} {l:6s}: {kp:>4}/{tot:>4}  ({100*kp/max(tot,1):.1f}%)")

    (DM_DIR / "pass" / "summary.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote -> {PASS_DIR}/{{train,val}}.jsonl + summary.json")


if __name__ == "__main__":
    main()
