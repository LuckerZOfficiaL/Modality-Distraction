r"""Cross-backbone agreement on flip destinations, per placebo arm (rebuttal asset).

Question (raised by the user, 2026-08-17): "caption-induced answer" -- is the flip destination
actually determined by the caption, or is it just whatever wrong option the model happened to
pick? Test: among items where >=2 backbones flip (V-correct -> VT-wrong), how often does a random
pair of flipping backbones choose the SAME wrong option, under each caption arm?

Result (2026-08-17): true caption 0.863, mismatch 0.671, scramble 0.709, neutral 0.714, uniform
baseline 0.333. Reading: destinations are item-determined across model families (far above
uniform even for content-free text -- every item has an intrinsically attractive distractor), and
the true caption pushes agreement further (0.71 -> 0.86) on top of raising the flip rate 2.2x.
So "the caption chooses the destination" is PARTLY true; the defensible in-paper claim is a
confident late commitment to a single wrong option (fig:lens), which is what the draft now says.

    python scripts/126_flip_destination_agreement.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

DR = Path(__file__).resolve().parent.parent / "data"
TAGS = ["qwen7b", "internvl", "qwen3b", "llavaov", "next", "qwen2b", "llava15"]


def load(p):
    seen = {}
    for l in Path(p).read_text().splitlines():
        if l.strip():
            r = json.loads(l); seen.setdefault(r["candidate_id"], r)
    return seen


def main():
    base = {t: load(DR / f"eval/t8/{t}_w0.0.jsonl") for t in TAGS}
    plc = {t: load(DR / f"eval/t8/placebo_{t}.jsonl") for t in TAGS}
    items = [c for c, r in base["qwen7b"].items() if r["label"] == "vision"]

    def agreement(getter):
        same = tot = multi = 0
        for c in items:
            dest = []
            for t in TAGS:
                rb = base[t].get(c)
                if rb is None or rb["pred_v"] != rb["correct_index"]:
                    continue
                d = getter(t, c, rb)
                if d is not None and d != rb["correct_index"]:
                    dest.append(d)
            if len(dest) >= 2:
                multi += 1
                for a, b in itertools.combinations(dest, 2):
                    tot += 1; same += int(a == b)
        return same / tot if tot else float("nan"), multi, tot

    out = {"true": agreement(lambda t, c, rb: rb["pred_vt"])}
    for arm in ("mismatch", "scramble", "neutral"):
        out[arm] = agreement(
            lambda t, c, rb, arm=arm: (plc[t].get(f"{arm}::{c}") or {}).get("pred_vt"))
    print(f"{'arm':10}{'pair-agreement':>15}{'items>=2':>10}{'pairs':>8}   (uniform = 0.333)")
    for k, (a, n, p) in out.items():
        print(f"{k:10}{a:>15.3f}{n:>10}{p:>8}")
    art = DR / "diagnostics/flip_destination_agreement.json"
    art.write_text(json.dumps({k: {"agreement": v[0], "items": v[1], "pairs": v[2]}
                               for k, v in out.items()}, indent=2))
    print(f"-> {art}")


if __name__ == "__main__":
    main()
