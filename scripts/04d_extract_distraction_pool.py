"""Step 4d: aggregate Qwen 3-pass predictions across all batches and emit a
unified distraction pool for steering eval.

Distraction events are Qwen-confirmed cases where one modality alone suffices
but the joint V+T fails:
  - v_distracted (V✓ ∧ VT✗): adding caption distracts Qwen from a question
    Qwen could solve from the image alone. → steer toward V to recover.
  - t_distracted (T✓ ∧ VT✗): adding image distracts Qwen from a caption-only
    answerable question. → steer toward T to recover.

For each event we re-emit the candidates.jsonl row plus:
  - label = "vision" (for v_distracted) or "text" (for t_distracted), so the
    sign convention of steering vectors (α>0 → V; α<0 → T) lines up with the
    direction that should help recovery — same as the original keep rule.
  - distraction_kind ∈ {v_distracted, t_distracted}
  - qwen_pred_vt, qwen_pred_v, qwen_pred_t — audit trail
  - __split = "test" — distraction pool is eval-only, never used to fit
    probes or steering vectors (those were already fit on oracle survivors).

Usage:
  python scripts/04d_extract_distraction_pool.py \
      --dm-dirs data/dm/gemini/batch1 data/dm/sonnet_hardT/batch6 \
                data/dm/sonnet/batch1 data/dm/sonnet/batch2 ... \
      --out data/dm/distraction_pool.jsonl

If --dm-dirs is omitted, the script auto-discovers every directory containing
both `candidates.jsonl` and `pass_qwen_candidates/predictions.jsonl` under
data/dm/ and data/dm/vistext/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def discover_dm_dirs(roots: list[Path]) -> list[Path]:
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for cand in root.rglob("candidates.jsonl"):
            d = cand.parent
            if (d / "pass_qwen_candidates" / "predictions.jsonl").is_file():
                found.append(d)
    return sorted(set(found))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dm-dirs", nargs="*", default=None,
                    help="paths to dm dirs; auto-discover if omitted")
    ap.add_argument("--auto-roots", nargs="*", default=["data/dm", "data/dm/vistext"],
                    help="roots to search when --dm-dirs is omitted")
    ap.add_argument("--out", default="data/dm/distraction_pool.jsonl")
    args = ap.parse_args()

    if args.dm_dirs:
        dm_dirs = [Path(d) for d in args.dm_dirs]
    else:
        dm_dirs = discover_dm_dirs([Path(r) for r in args.auto_roots])
    if not dm_dirs:
        print("no dm dirs found")
        return
    print(f"scanning {len(dm_dirs)} dm dir(s):")
    for d in dm_dirs:
        print(f"  {d}")

    out_rows: list[dict] = []
    per_dir_stats: list[tuple[str, int, int, int]] = []
    seen_cids: set[str] = set()  # dedupe across batches

    for d in dm_dirs:
        cands = {}
        for l in (d / "candidates.jsonl").read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            cands[r["candidate_id"]] = r
        preds = {}
        for l in (d / "pass_qwen_candidates" / "predictions.jsonl").read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            preds[r["candidate_id"]] = r

        n_v_dist = n_t_dist = 0
        for cid, p in preds.items():
            if cid not in cands:
                continue
            if cid in seen_cids:
                continue
            ci = p["correct_index"]
            vt_ok = p["vt"] == ci
            v_ok  = p["v"]  == ci
            t_ok  = p["t"]  == ci
            if vt_ok:
                continue  # not a distraction event
            kind = None
            if v_ok and not t_ok:
                kind, label = "v_distracted", "vision"
                n_v_dist += 1
            elif t_ok and not v_ok:
                kind, label = "t_distracted", "text"
                n_t_dist += 1
            elif v_ok and t_ok:
                # both alone right, joint fails — ambiguous direction.
                # Default to v_distracted (treat as vision); skip if you want
                # to study this asymmetry separately.
                kind, label = "v_distracted", "vision"
                n_v_dist += 1
            else:
                continue
            c = cands[cid]
            out_rows.append({
                "candidate_id": cid,
                "seed_id": c.get("seed_id"),
                "image_path": c["image_path"],
                "original_caption": c.get("original_caption"),
                "label": label,
                "question": c["question"],
                "options": c["options"],
                "correct_index": c["correct_index"],
                "caption_for_filter": c["caption_for_filter"],
                "rationale": c.get("rationale", ""),
                "distraction_kind": kind,
                "qwen_pred_vt": p["vt"],
                "qwen_pred_v":  p["v"],
                "qwen_pred_t":  p["t"],
                "__split": "test",
                "source_dm_dir": str(d),
            })
            seen_cids.add(cid)
        per_dir_stats.append((str(d), n_v_dist, n_t_dist, n_v_dist + n_t_dist))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'dm_dir':<60} {'V-dist':>8} {'T-dist':>8} {'total':>8}")
    print("-" * 90)
    for d, nv, nt, n in per_dir_stats:
        print(f"{d:<60} {nv:>8} {nt:>8} {n:>8}")
    grand_v = sum(r["distraction_kind"] == "v_distracted" for r in out_rows)
    grand_t = sum(r["distraction_kind"] == "t_distracted" for r in out_rows)
    print("-" * 90)
    print(f"{'TOTAL (deduped)':<60} {grand_v:>8} {grand_t:>8} {len(out_rows):>8}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
