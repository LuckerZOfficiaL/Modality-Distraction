r"""What the certification gate actually removes -- exactly, from the stored oracle answers.

Every generation batch keeps the oracle's three answers per candidate under work/ans_{v,t,vt}/,
so the gate's decision is fully reconstructible for the ~11k candidates it rejected. No GPU, no
model calls: this is the composition of the benchmark a pipeline would ship if it generated items
and trusted the generator's intended grounding label without verifying it.

Reports, for intended-vision and intended-text candidates separately, the share of rejects that
are (a) not answerable from the grounding modality at all, (b) answerable from the OFF modality
(the label is simply wrong), (c) not answerable even with both inputs, and the overlaps.

    python scripts/119_gate_decomposition.py
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
DOM = {"gemini": "dci", "vistext": "vistext", "semart": "semart", "roco": "roco"}
SKIP = ("hardT", "smoke", "sonnet/batch_1-4")


def batches():
    for c in sorted(glob.glob(str(DR / "dm/**/candidates.jsonl"), recursive=True)):
        d = Path(c).parent
        if any(x in str(d) for x in SKIP) or d.name == "batch2" or "multidomain_v1" in str(d):
            continue
        if not (d / "work/ans_v/results.jsonl").exists():
            continue
        dom = next((k for k in DOM if k in str(d)), None)
        if dom:
            yield d, DOM[dom]


def answers(d, cond):
    p = d / f"work/ans_{cond}/results.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["candidate_id"]] = r.get("predicted_index")
    return out


def main():
    kept = {json.loads(l)["candidate_id"]
            for l in (DR / "dm/multidomain_v1/dm_all.jsonl").read_text().splitlines() if l.strip()}
    rows, n_batches = [], 0
    for d, dom in batches():
        n_batches += 1
        av, at, avt = answers(d, "v"), answers(d, "t"), answers(d, "vt")
        for line in (d / "candidates.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["candidate_id"]
            if cid not in av or cid not in at or cid not in avt:
                continue
            ci = r["correct_index"]
            rows.append({"cid": cid, "dom": dom, "label": r.get("label"), "kept": cid in kept,
                         "V": av[cid] == ci, "T": at[cid] == ci, "VT": avt[cid] == ci})
    byid = {}
    for r in rows:
        byid.setdefault(r["cid"], r)
    rows = list(byid.values())
    print(f"batches with stored oracle answers: {n_batches}   candidates reconstructed: {len(rows)}")

    # Coverage is NOT uniform: SemArt and ROCO kept every per-condition answer, DCI kept only a
    # sample of batch1's, and the VisText batches kept none. Rejected candidates carry no pass
    # flags (only survivors do), so the exact decomposition is available on the complete domains
    # and DCI serves as a consistency sample. Report it that way rather than pooling silently.
    gen = collections.Counter(r["dom"] for r in rows)
    print("\ncandidates with a full three-condition oracle record, by domain:")
    for d in ("dci", "vistext", "semart", "roco"):
        print(f"   {d:9} {gen.get(d, 0):6}")
    # Validity gate on the reconstruction itself. "Not in dm_all" is only the same thing as
    # "oracle-rejected" where nothing else removed items downstream. DCI's batch also ran a Qwen
    # behavioral filter (pass_qwen*/) and was subsampled, so a large share of its non-kept
    # candidates carry the PASS signature -- there, this reconstruction measures the wrong thing.
    # Detect that per domain and drop any domain that fails, rather than reporting it as a check.
    print("\nreconstruction validity -- share of non-kept candidates that carry the PASS signature")
    print("(must be ~0 for 'not kept' to mean 'oracle-rejected'):")
    COMPLETE = set()
    for d in ("dci", "vistext", "semart", "roco"):
        rej = [r for r in rows if r["dom"] == d and not r["kept"]]
        if not rej:
            continue
        bad = sum(1 for r in rej
                  if (r["V"] and not r["T"] and r["VT"] and r["label"] == "vision")
                  or (r["T"] and not r["V"] and r["VT"] and r["label"] == "text"))
        frac = bad / len(rej)
        ok = frac < 0.02
        print(f"   {d:9} {100*frac:5.1f}%   {'OK' if ok else 'CONTAMINATED -> dropped'}")
        if ok:
            COMPLETE.add(d)
    print(f"   -> exact decomposition below uses {sorted(COMPLETE)} "
          f"({sum(gen.get(d, 0) for d in COMPLETE)} candidates)")

    def block(rs, title):
        print(f"\n{'#' * 76}\n# {title}\n{'#' * 76}")
        for lab, ground, off in (("vision", "V", "T"), ("text", "T", "V")):
            sub = [r for r in rs if r["label"] == lab]
            rej = [r for r in sub if not r["kept"]]
            if not sub or not rej:
                continue
            n, m = len(rej), len(sub)
            print(f"  intended {lab.upper():6} n={m:5} ({m - n} certified, {n} rejected)"
                  f"   ungated pool would be {100*sum(1 for r in sub if r[off])/m:5.1f}% off-modality answerable,"
                  f" {100*sum(1 for r in sub if r[ground] and not r[off])/m:5.1f}% genuinely grounded")

    block([r for r in rows if r["dom"] in COMPLETE],
          f"{' + '.join(sorted(d.upper() for d in COMPLETE))} (reconstruction valid)")
    rows = [r for r in rows if r["dom"] in COMPLETE]

    for lab, ground, off in (("vision", "V", "T"), ("text", "T", "V")):
        sub = [r for r in rows if r["label"] == lab]
        rej = [r for r in sub if not r["kept"]]
        keep = [r for r in sub if r["kept"]]
        if not sub:
            continue
        print(f"\n=== intended {lab.upper()}-grounded candidates: {len(sub)} "
              f"({len(keep)} certified, {len(rej)} rejected) ===")
        n = len(rej)
        no_ground = sum(1 for r in rej if not r[ground])
        off_leak = sum(1 for r in rej if r[off])
        no_vt = sum(1 for r in rej if not r["VT"])
        only_leak = sum(1 for r in rej if r[ground] and r[off])
        print(f"  of the REJECTED ones:")
        print(f"    not answerable from the grounding modality   {no_ground:6} ({100*no_ground/n:5.1f}%)")
        print(f"    answerable from the OFF modality (mislabel)  {off_leak:6} ({100*off_leak/n:5.1f}%)")
        print(f"      ...of which grounding also works (pure leak){only_leak:6} ({100*only_leak/n:5.1f}%)")
        print(f"    not answerable even from BOTH inputs         {no_vt:6} ({100*no_vt/n:5.1f}%)")
        # what an ungated benchmark of this label would look like
        allc = sub
        m = len(allc)
        print(f"  if the gate were skipped, this label's pool would be {m} items, of which:")
        print(f"    genuinely single-modality grounded           "
              f"{sum(1 for r in allc if r[ground] and not r[off]):6} ({100*sum(1 for r in allc if r[ground] and not r[off])/m:5.1f}%)")
        print(f"    off-modality answerable (label is wrong)     "
              f"{sum(1 for r in allc if r[off]):6} ({100*sum(1 for r in allc if r[off])/m:5.1f}%)")
        print(f"    unanswerable from the grounding modality     "
              f"{sum(1 for r in allc if not r[ground]):6} ({100*sum(1 for r in allc if not r[ground])/m:5.1f}%)")

    out = DR / "diagnostics/gate_decomposition.json"
    agg = {}
    for lab, ground, off in (("vision", "V", "T"), ("text", "T", "V")):
        sub = [r for r in rows if r["label"] == lab]
        if not sub:
            continue
        rej = [r for r in sub if not r["kept"]]
        agg[lab] = {
            "n_total": len(sub), "n_certified": sum(1 for r in sub if r["kept"]), "n_rejected": len(rej),
            "rejected_no_grounding": sum(1 for r in rej if not r[ground]) / max(len(rej), 1),
            "rejected_offmodality": sum(1 for r in rej if r[off]) / max(len(rej), 1),
            "rejected_no_vt": sum(1 for r in rej if not r["VT"]) / max(len(rej), 1),
            "ungated_clean_frac": sum(1 for r in sub if r[ground] and not r[off]) / len(sub),
            "ungated_offmodality_frac": sum(1 for r in sub if r[off]) / len(sub),
            "ungated_no_grounding_frac": sum(1 for r in sub if not r[ground]) / len(sub),
        }
    agg["per_domain_reconstructed"] = dict(collections.Counter(r["dom"] for r in rows))
    out.write_text(json.dumps(agg, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
