r"""Placebo-caption ladder: is v-distraction specific to the certified-irrelevant caption, or do
answer flips appear under ANY added text?

The paper measures v-distraction as flips caused by the item's true (certified answer-irrelevant)
caption. Nothing yet separates that from generic instability under prompt growth. Three sibling
arms per MoGround vision item, graded by how much of the true caption's structure they keep:

  P1 mismatch  another item's caption from the SAME domain (real, fluent, topically wrong).
               Within-domain derangement: items sorted by caption word count, cyclic shift by one,
               so donor lengths are matched almost exactly and no item keeps its own caption.
  P2 scramble  the item's OWN caption with word order shuffled (identical tokens, no propositions)
  P3 neutral   a fixed content-free fluent text, tiled/truncated to the item's caption word count

P0 (the true caption) is already on disk as the base runs. All arms are scored on the SAME
denominator: items the backbone answers correctly from V-only in its base run, so the comparison
is paired at the item level. Only the VT condition needs running (scripts/behavioral_eval.py --conds VT).

    python scripts/build_placebo_pools.py
        -> data/dm/placebo_v1/splits/test.jsonl (3 arms x vision items) + configs/placebo_v1.yaml
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
OUT = DR / "dm/placebo_v1"

# Fluent, content-free, modality-agnostic filler; tiled to the needed word count. Deliberately
# avoids objects, colours, counts, places -- anything an option could match.
NEUTRAL = ("Conditions at the time were considered fairly ordinary overall, and observers "
           "generally agreed that the situation developed in a way most would describe as "
           "typical for the period in question, with nothing unusual reported.").split()


def main():
    rng = random.Random(0)
    rows = []
    for line in (DR / "dm/multidomain_v1/dm_all.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["label"] == "vision" and Path(r["image_path"]).exists():
                rows.append(r)
    print(f"vision items with image on disk: {len(rows)}")

    # P1: within-domain, length-sorted cyclic derangement
    donor = {}
    for dom in sorted({r["source"] for r in rows}):
        grp = sorted([r for r in rows if r["source"] == dom],
                     key=lambda r: (len(str(r["caption_for_filter"]).split()), r["candidate_id"]))
        for i, r in enumerate(grp):
            j = (i + 1) % len(grp)
            # skip over textually identical captions (shared seeds); bounded walk
            steps = 0
            while grp[j]["caption_for_filter"] == r["caption_for_filter"] and steps < len(grp):
                j = (j + 1) % len(grp); steps += 1
            donor[r["candidate_id"]] = grp[j]["caption_for_filter"]

    pool = []
    for r in rows:
        true_cap = str(r["caption_for_filter"])
        n_words = max(len(true_cap.split()), 1)
        sc = true_cap.split()
        rng.shuffle(sc)
        arms = {
            "mismatch": donor[r["candidate_id"]],
            "scramble": " ".join(sc),
            "neutral": " ".join((NEUTRAL * (n_words // len(NEUTRAL) + 1))[:n_words]),
        }
        for arm, cap in arms.items():
            pool.append({"candidate_id": f"{arm}::{r['candidate_id']}",
                         "question": r["question"], "options": r["options"],
                         "correct_index": int(r["correct_index"]),
                         "caption_for_filter": cap, "image_path": r["image_path"],
                         "label": "vision", "source": r["source"]})
    rng.shuffle(pool)
    (OUT / "splits").mkdir(parents=True, exist_ok=True)
    with (OUT / "splits/test.jsonl").open("w") as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")
    (ROOT / "configs/placebo_v1.yaml").write_text(f"""# Placebo-caption ladder (scripts/build_placebo_pools.py). Internal; VT condition only (scripts/behavioral_eval.py --conds VT).
dm:
  split: [0.6, 0.2, 0.2]
model:
  name: Qwen/Qwen2.5-VL-3B-Instruct
paths:
  data_root: {ROOT}/data
  dm_dir: data/dm/placebo_v1
  dm_images: {ROOT}/data/dm/images
  runs: {ROOT}/runs
seed: 0
""")
    # sanity: mismatch donor lengths track the true caption's
    import statistics as st
    diffs = [abs(len(donor[r["candidate_id"]].split()) - len(str(r["caption_for_filter"]).split()))
             for r in rows]
    print(f"wrote {OUT}/splits/test.jsonl  ({len(pool)} rows = {len(rows)} items x 3 arms)")
    print(f"mismatch length diff (words): median {st.median(diffs)}  p90 {sorted(diffs)[int(.9*len(diffs))]}")


if __name__ == "__main__":
    main()
