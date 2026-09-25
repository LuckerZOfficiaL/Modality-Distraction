"""Score the free-form pilot and compare it to the MCQ numbers on the SAME items.

Two independent scorers, because the interesting quantity (did the caption corrupt the answer?)
must not depend on one matching rule:

  strict   -- the gold option's content words all appear in the generated answer
  project  -- the answer is embedded and projected onto the item's own option set
              (all-mpnet-base-v2), turning a free-form answer into an induced choice, which
              makes free-form and MCQ directly comparable on the same estimator

Reported per model:
  * V-only / V+T / T-only free-form accuracy
  * free-form v-distraction, P(V+T wrong | V-only correct), the paper's own-set estimator
  * MCQ v-distraction on the identical item subset, read from data/eval/t8/
  * PROVENANCE: of the items that flip, how often the wrong free-form answer names a distractor
    option that the caption actually mentions ("planted"), versus one it does not. MCQ cannot
    separate these two, free-form can, and that is the point of the pilot.

    python scripts/freeform_score.py --key qwen2.5-vl-3b --mcq qwen3b
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STOP = {"a", "an", "the", "is", "are", "of", "in", "on", "at", "to", "it", "this", "that",
        "there", "and", "or", "be", "as", "with", "for", "its", "his", "her", "their",
        "approximately", "about", "around", "roughly"}
PUNCT = str.maketrans("", "", string.punctuation.replace(".", ""))

# Free-form answers say "2" and "4000" where the option list says "Two" and "4,300". Without this
# the scorer would measure its own matching rule instead of the model, so numbers are normalised
# on both sides and numeric items are matched by value rather than by string.
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}
REFUSE = re.compile(r"not (provided|mentioned|specified|stated|given|possible|clear|available)"
                    r"|cannot (be )?(determine|tell|answer)|can't (determine|tell)|no information"
                    r"|unable to|does not (provide|mention|specify|state)|insufficient",
                    re.IGNORECASE)


def to_num(s):
    """Parse '4,300', '4300', '12', 'twenty four', '1.5 million' -> float, else None."""
    t = s.lower().replace(",", "").strip()
    m = re.fullmatch(r"[^0-9.\-]*(-?\d+(?:\.\d+)?)\s*(hundred|thousand|million|billion)?[^0-9]*", t)
    if m:
        v = float(m.group(1))
        return v * SCALES[m.group(2)] if m.group(2) else v
    toks = [w for w in re.split(r"[\s-]+", t) if w]
    if not toks or not all(w in UNITS or w in SCALES for w in toks):
        return None
    total = cur = 0.0
    for w in toks:
        if w in UNITS:
            cur += UNITS[w]
        else:
            sc = SCALES[w]
            cur = (cur or 1) * sc
            if sc >= 1000:
                total += cur; cur = 0.0
    return total + cur


def norm(s):
    out = []
    for w in s.lower().translate(PUNCT).split():
        if w in STOP:
            continue
        v = to_num(w)
        out.append(str(int(v)) if v is not None and float(v).is_integer() else w)
    return " ".join(out)


def words(s):
    return set(norm(s).split())


def strict_match(ans, opt):
    """All of the option's content words present in the answer (order-free containment).

    Numbers are compared by value, spelled out or not, so 'Twenty four' matches '24'.
    """
    a, o = to_num(ans), to_num(opt)
    if a is not None and o is not None:
        return a == o
    ow = words(opt)
    return bool(ow) and ow <= words(ans)


def score_strict(ans, options, gold):
    if REFUSE.search(ans or ""):
        return -1
    hits = [i for i, o in enumerate(options) if strict_match(ans, o)]
    return hits[0] if len(hits) == 1 else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="stem under data/eval/freeform/")
    ap.add_argument("--mcq", required=True, help="tag under data/eval/t8/ for the same backbone")
    ap.add_argument("--sim-floor", type=float, default=0.25,
                    help="below this cosine, the projection abstains")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (ROOT / f"data/eval/freeform/{args.key}.jsonl").read_text().splitlines() if l.strip()]
    seen = {}
    for r in rows:
        seen.setdefault(r["candidate_id"], r)
    rows = list(seen.values())
    print(f"{args.key}: {len(rows)} certified vision items\n")

    # ---- projection scorer -----------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    texts, spans = [], []
    for r in rows:
        for cond in ("gen_v", "gen_vt", "gen_t"):
            spans.append((r["candidate_id"], cond, len(texts))); texts.append(r[cond] or "")
        r["_optoff"] = len(texts); texts.extend(r["options"])
    E = enc.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

    for r in rows:
        O = E[r["_optoff"]:r["_optoff"] + len(r["options"])]
        onum = [to_num(o) for o in r["options"]]
        numeric_item = all(x is not None for x in onum)
        for cond in ("gen_v", "gen_vt", "gen_t"):
            idx = next(i for cid, c, i in spans if cid == r["candidate_id"] and c == cond)
            ans = r[cond] or ""
            if REFUSE.search(ans):
                r[f"proj_{cond}"] = -1
            else:
                anum = to_num(ans)
                if numeric_item and anum is not None:
                    # nearest option by relative distance, which is how a human would grade
                    # "4000" against {700, 11000, 4300, 200}
                    d = [abs(anum - x) / max(abs(x), 1e-9) for x in onum]
                    r[f"proj_{cond}"] = int(np.argmin(d))
                else:
                    sims = O @ E[idx]
                    j = int(sims.argmax())
                    r[f"proj_{cond}"] = j if float(sims[j]) >= args.sim_floor else -1
            r[f"strict_{cond}"] = score_strict(ans, r["options"], r["correct_index"])

    # ---- accuracies and distraction --------------------------------------------------------
    def acc(pred_key):
        return float(np.mean([r[pred_key] == r["correct_index"] for r in rows]))

    def vdistr(pred_v, pred_vt):
        s = [r for r in rows if r[pred_v] == r["correct_index"]]
        return (float(np.mean([r[pred_vt] != r["correct_index"] for r in s])), len(s))

    out = {"n_items": len(rows)}
    for tag, pre in (("projection", "proj"), ("strict", "strict")):
        v, vt, t = acc(f"{pre}_gen_v"), acc(f"{pre}_gen_vt"), acc(f"{pre}_gen_t")
        d, n = vdistr(f"{pre}_gen_v", f"{pre}_gen_vt")
        nom = float(np.mean([r[f"{pre}_gen_v"] == -1 for r in rows]))
        print(f"[{tag}] V-only {v:.3f}   V+T {vt:.3f}   T-only {t:.3f}   "
              f"no-match(V) {nom:.3f}")
        print(f"[{tag}] free-form v-distraction {d:.3f}  (conditioned on {n} V-correct items)")
        out[tag] = {"acc_v": v, "acc_vt": vt, "acc_t": t, "v_distraction": d, "n_cond": n,
                    "no_match_rate_v": nom}

    # ---- the same items under MCQ ----------------------------------------------------------
    ids = {r["candidate_id"] for r in rows}
    mcq, seen_m = {}, set()
    for l in (ROOT / f"data/eval/t8/{args.mcq}_w0.0.jsonl").read_text().splitlines():
        if l.strip():
            m = json.loads(l)
            if m["candidate_id"] in ids and m["candidate_id"] not in seen_m:
                seen_m.add(m["candidate_id"]); mcq[m["candidate_id"]] = m
    s = [m for m in mcq.values() if m["pred_v"] == m["correct_index"]]
    mcq_d = float(np.mean([m["pred_vt"] != m["correct_index"] for m in s]))
    mcq_v = float(np.mean([m["pred_v"] == m["correct_index"] for m in mcq.values()]))
    print(f"\n[MCQ, same {len(mcq)} items] V-only {mcq_v:.3f}   "
          f"v-distraction {mcq_d:.3f}  (conditioned on {len(s)} V-correct items)")
    out["mcq"] = {"n": len(mcq), "acc_v": mcq_v, "v_distraction": mcq_d, "n_cond": len(s)}

    # ---- PAIRED comparison ------------------------------------------------------------------
    # Free-form and MCQ condition on different item sets (free-form V-only accuracy is lower,
    # because there are no options to choose from), so the headline comparison must be run on the
    # items BOTH formats solve from the image alone. Same items, same model, only the answer
    # format differs, with an exact McNemar on the discordant pairs.
    from scipy.stats import binomtest
    common = [r for r in rows if r["candidate_id"] in mcq
              and r["proj_gen_v"] == r["correct_index"]
              and mcq[r["candidate_id"]]["pred_v"] == mcq[r["candidate_id"]]["correct_index"]]
    ff_flip = [r["proj_gen_vt"] != r["correct_index"] for r in common]
    mc_flip = [mcq[r["candidate_id"]]["pred_vt"] != mcq[r["candidate_id"]]["correct_index"]
               for r in common]
    b = sum(1 for f, m in zip(ff_flip, mc_flip) if f and not m)     # free-form only
    c = sum(1 for f, m in zip(ff_flip, mc_flip) if m and not f)     # MCQ only
    pv = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    print(f"\n[paired, {len(common)} items solved from the image in BOTH formats] "
          f"free-form flips {np.mean(ff_flip):.3f}  vs  MCQ flips {np.mean(mc_flip):.3f}")
    print(f"      discordant: free-form-only {b}, MCQ-only {c}, exact McNemar p={pv:.4f}")
    out["paired"] = {"n": len(common), "ff": float(np.mean(ff_flip)), "mcq": float(np.mean(mc_flip)),
                     "ff_only": b, "mcq_only": c, "p": float(pv)}

    # ---- provenance: does the flipped answer name an option the CAPTION mentions? -----------
    # "planted" = the distractor's content words all appear in the caption. If flips land on
    # planted distractors far more than on unplanted ones, the caption's content is provably
    # inside the generated answer, which is the measurement MCQ cannot make.
    for pre in ("proj", "strict"):
        flip = [r for r in rows
                if r[f"{pre}_gen_v"] == r["correct_index"] and r[f"{pre}_gen_vt"] != r["correct_index"]]
        planted_hit = unplanted_hit = nomatch = 0
        planted_avail = unplanted_avail = 0
        for r in flip:
            cap = r["caption"]
            dis = [i for i in range(len(r["options"])) if i != r["correct_index"]]
            planted = [i for i in dis if strict_match(cap, r["options"][i])]
            planted_avail += len(planted); unplanted_avail += len(dis) - len(planted)
            j = r[f"{pre}_gen_vt"]
            if j == -1:
                nomatch += 1
            elif j in planted:
                planted_hit += 1
            else:
                unplanted_hit += 1
        n = max(len(flip), 1)
        print(f"\n[{pre}] provenance over {len(flip)} flipped items: "
              f"lands on a caption-planted distractor {planted_hit} ({planted_hit / n:.2f}), "
              f"on an unplanted distractor {unplanted_hit} ({unplanted_hit / n:.2f}), "
              f"no match {nomatch} ({nomatch / n:.2f})")
        if planted_avail and unplanted_avail:
            print(f"      per-option rate: planted {planted_hit / planted_avail:.3f} "
                  f"vs unplanted {unplanted_hit / unplanted_avail:.3f} "
                  f"(availability-normalised, {planted_avail} vs {unplanted_avail} options)")
        out[f"provenance_{pre}"] = {
            "n_flip": len(flip), "planted": planted_hit, "unplanted": unplanted_hit,
            "no_match": nomatch, "planted_avail": planted_avail, "unplanted_avail": unplanted_avail}

    fp = ROOT / f"data/diagnostics/freeform_pilot_{args.key}.json"
    fp.write_text(json.dumps(out, indent=1))
    print("\nwrote", fp)


if __name__ == "__main__":
    main()
