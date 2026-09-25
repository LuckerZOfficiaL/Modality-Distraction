r"""Assemble the public release from the working data, and snapshot the internal originals.

Produces two trees:

  data/release/          what ships: MoGround (+ splits), MoGround-Human, the audited assembled
                         pool, the caption-irrelevance judgments, and the 250-item human audit of
                         single-modality answerability. Item records carry only dataset fields:
                         identifiers, the image, the caption, the question, the options, the
                         answer, the grounding label and (for MoGround) the three-condition
                         certificate. Internal workflow bookkeeping is dropped.

  data/internal_backup/  byte-identical copies of the source files, so nothing is lost.

Field policy for the released records:
  keep    candidate_id, seed_id, domain, label, image, caption, original_caption, question,
          options, correct_index, rationale; MoGround also keeps pass_v / pass_t / pass_vt
          (the behavioural certificate) and its split; the assembled pool keeps its source
          benchmark and retrieval score
  drop    subbatch, validation_regime, the authoring bookkeeping on the human set, and the
          human audit's per-rejection `reason` (the released rate counts every rejection alike,
          so the breakdown is not part of any claim; the full file is kept in internal_backup/)
  rename  MoGround-Human's overlap_before / overlap_after -> distractor_overlap_source /
          distractor_overlap_authored, which is what they measure (source caption vs authored
          caption, against the wrong option the caption targets)
  rewrite image paths to be repo-relative

    python scripts/107_build_release.py
"""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
REL = DR / "release"
BAK = DR / "internal_backup"

SOURCES = [
    DR / "dm/multidomain_v1/splits/train.jsonl",
    DR / "dm/multidomain_v1/splits/val.jsonl",
    DR / "dm/multidomain_v1/splits/test.jsonl",
    DR / "dm/human_moground/frozen_20260801.jsonl",
    DR / "dm/human_moground/conflict_audit.json",
    DR / "dm/merged_aokvqa_racehigh/dm_all.jsonl",
    DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json",
    DR / "diagnostics/assembled_caption_certification.json",
    DR / "moground_validation/validations.jsonl",
]


def rel_img(p):
    p = str(p)
    return p.split("moground/", 1)[1] if "moground/" in p else p


def opts(o):
    return ast.literal_eval(o) if isinstance(o, str) else o


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"  {path.relative_to(DR)}  ({len(rows)} items)")


def base(r, domain):
    return dict(candidate_id=r["candidate_id"], seed_id=r.get("seed_id"), domain=domain,
                label=r["label"], image=rel_img(r["image_path"]),
                caption=r.get("caption_for_filter") or "", original_caption=r.get("original_caption") or "",
                question=r["question"], options=opts(r["options"]),
                correct_index=int(r["correct_index"]), rationale=r.get("rationale") or "")


def main():
    # ---- 0. snapshot the internal originals ---------------------------------------------------
    BAK.mkdir(parents=True, exist_ok=True)
    print("internal backup:")
    for src in SOURCES:
        if src.exists():
            dst = BAK / src.relative_to(DR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  {dst.relative_to(DR)}")

    if REL.exists():
        shutil.rmtree(REL)

    # ---- 1. MoGround, with its certificate and split ------------------------------------------
    print("\nrelease:")
    allm = []
    for sp in ("train", "val", "test"):
        rows = []
        for r in jl(DR / f"dm/multidomain_v1/splits/{sp}.jsonl"):
            d = base(r, r.get("source"))
            d.update(split=sp, pass_v=r.get("pass_v"), pass_t=r.get("pass_t"), pass_vt=r.get("pass_vt"))
            rows.append(d)
        write_jsonl(REL / f"moground/splits/{sp}.jsonl", rows)
        allm += rows
    write_jsonl(REL / "moground/items.jsonl", allm)

    # ---- 2. MoGround-Human --------------------------------------------------------------------
    hrows = []
    for r in jl(DR / "dm/human_moground/frozen_20260801.jsonl"):
        d = base(r, r.get("source"))
        for src_k, dst_k in (("overlap_before", "distractor_overlap_source"),
                             ("overlap_after", "distractor_overlap_authored")):
            v = r.get(src_k)
            if v not in (None, ""):
                d[dst_k] = round(float(v), 4)
        hrows.append(d)
    write_jsonl(REL / "moground_human/items.jsonl", hrows)
    shutil.copy2(DR / "dm/human_moground/conflict_audit.json",
                 REL / "moground_human/caption_audit.json")
    print("  moground_human/caption_audit.json")

    # ---- 3. assembled pool, post-audit --------------------------------------------------------
    cert = json.loads((DR / "diagnostics/assembled_caption_certification.json").read_text())
    viol = set(cert["violations"])
    split = json.loads((DR / "dm/merged_aokvqa_racehigh/assembled_dev_heldout_split.json").read_text())
    half = {c: "dev" for c in split["dev"]}
    half.update({c: "heldout" for c in split["heldout"]})
    seen, arows = set(), []
    for r in jl(DR / "dm/merged_aokvqa_racehigh/dm_all.jsonl"):
        cid = r["candidate_id"]
        if cid in seen or cid in viol:
            continue
        seen.add(cid)
        d = base(r, "aokvqa" if r["label"] == "vision" else "race_high")
        # in this pool the "original" caption is the retrieved image's own caption, i.e. a property
        # of the cross-modal distractor rather than a source caption: name it for what it is
        d["distractor_image_caption"] = d.pop("original_caption")
        d.update(half=half.get(cid), source_benchmark=r.get("source_text_qa") or "aokvqa",
                 retrieval_score=(round(float(r["retrieval_score"]), 4)
                                  if r.get("retrieval_score") not in (None, "") else None))
        arows.append(d)
    write_jsonl(REL / "assembled/items.jsonl", arows)
    (REL / "assembled/caption_irrelevance_audit.json").write_text(json.dumps(
        {"audited": cert["n"], "verdict_counts": cert["counts"],
         "dropped_candidate_ids": sorted(viol),
         "note": "Vision items only: the retrieved caption was judged for whether it asserts or "
                 "contradicts an option. Items listed here are excluded from items.jsonl."},
        indent=2))
    print("  assembled/caption_irrelevance_audit.json")

    # ---- 3b. human audit labels ---------------------------------------------------------------
    # The paper cites a 250-item hand audit; ship the labels so the claim is checkable, along with
    # the notebook and seed that reproduce the sample and its order.
    vp = DR / "moground_validation/validations.jsonl"
    audit_rows = []
    if vp.exists():
        latest = {}
        for r in jl(vp):                       # last verdict per item wins, as in the analysis
            latest[r["candidate_id"]] = r
        # `reason` stays internal: the released rate counts every rejection equally, so the
        # per-reason breakdown is not part of the claim. The full file is in internal_backup/.
        # The annotator's records carry no split, so recover it from the released splits.
        split_of = {r["candidate_id"]: r["split"] for r in allm}
        audit_rows = [dict(candidate_id=c, domain=r.get("domain"), modality=r.get("modality"),
                           split=split_of.get(c), verdict=r["verdict"], seed=r.get("seed"))
                      for c, r in latest.items()]
        write_jsonl(REL / "moground/human_audit.jsonl", audit_rows)
        miss = [r for r in audit_rows if r["split"] is None]
        if miss:
            print(f"  WARNING: {len(miss)} audited ids not found in the released splits")
        nb = DR / "moground_validation/validate.ipynb"
        if nb.exists():
            # ship the interface, not one session's state: clear outputs and execution counts
            j = json.loads(nb.read_text())
            for cell in j.get("cells", []):
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []; cell["execution_count"] = None
            (REL / "moground/human_audit_notebook.ipynb").write_text(json.dumps(j, indent=1))
            print("  moground/human_audit_notebook.ipynb")

    # ---- 4. README ----------------------------------------------------------------------------
    nv = sum(r["label"] == "vision" for r in arows)
    n_av = sum(r["modality"] == "vision" for r in audit_rows)
    n_at = sum(r["modality"] == "text" for r in audit_rows)
    (REL / "README.md").write_text(f"""# MoGround release

Three evaluation resources for studying **modality distraction**: a model answering an item
correctly from one modality alone, then wrongly once answer-irrelevant context from the other
modality is added.

## Contents

| path | items | what it is |
|---|---|---|
| `moground/items.jsonl` | {len(allm)} | oracle-certified single-modality-grounding items |
| `moground/splits/{{train,val,test}}.jsonl` | {len(allm)} | the fixed 60/20/20 split, stratified by domain and grounding |
| `moground/human_audit.jsonl` | 250 | hand audit of single-modality answerability, one label per sampled item |
| `moground/human_audit_notebook.ipynb` | — | the annotation interface; the seed inside reproduces the sample and its order |
| `moground_human/items.jsonl` | {len(hrows)} | hand-authored hard subset, captions written to tempt a wrong option |
| `moground_human/caption_audit.json` | 63 | per-item LLM judgement of whether a caption asserts an option |
| `assembled/items.jsonl` | {len(arows)} | companion pool ({nv} A-OKVQA vision / {len(arows)-nv} RACE-high text), post-audit |
| `assembled/caption_irrelevance_audit.json` | — | verdict counts and the candidate ids dropped by the audit |

## Item fields

`candidate_id`, `seed_id`, `domain`, `label` (`vision` / `text`), `image`, `caption`,
`original_caption`, `question`, `options`, `correct_index`, `rationale`.\n(In the assembled pool the retrieved image's own caption is given as\n`distractor_image_caption` instead of `original_caption`, since it describes the\ndistractor rather than the item's source.)

* **MoGround** adds `split` and the three-condition certificate `pass_v` / `pass_t` / `pass_vt`:
  the oracle's correctness with the image alone, the caption alone, and both. A vision-grounded
  item satisfies `pass_v and pass_vt and not pass_t`; a text-grounded item the mirror image.
* **MoGround-Human** adds `distractor_overlap_source` and `distractor_overlap_authored`: the
  lexical overlap between the wrong option a caption targets and, respectively, the original source
  caption and the authored one.
* **assembled** adds `half` (`dev` / `heldout`), `source_benchmark` and `retrieval_score`, the
  cosine similarity by which the cross-modal distractor was retrieved.

## Human audit

`moground/human_audit.jsonl` holds one record per audited item: `candidate_id`, `domain`,
`modality`, `split`, `seed`, and `verdict` (`yes` / `no`). An item passes only if both halves of
the certificate hold under human inspection: the intended modality answers it, and the other
modality does not. Rejections count against the pass rate whatever the reason, which makes the
released rate a lower bound on certificate quality.

The sample is a stratified random draw of {len(audit_rows)} items ({n_av} vision-grounded, {n_at} text-grounded)
across all four domains. `seed` is the permutation seed that fixes both the sample and the order in
which items were presented, so `human_audit_notebook.ipynb` reproduces the queue exactly.

## How to score distraction

Run each item three times: image only (V), context only (T), and both (V+T). Condition on the
items a model answers correctly from its grounding modality alone, then

```
v-distraction = P(V+T wrong | V-only correct)   over vision-grounded items
t-distraction = P(V+T wrong | T-only correct)   over text-grounded items
```

The conditioning is model-relative on purpose: each model is scored on the items it itself solves
from the grounded modality, so distraction always measures a capability the model demonstrably has
and loses.

## Notes

* `image` paths are repo-relative.
* Certification is operational: single-modality answerability holds relative to the oracles used,
  not as absolute ground truth. A cross-oracle check on a stratified 240-item sample puts
  independent-oracle agreement on single-modality answerability at 94%.
* The assembled pool ships post-audit; the dropped items are listed rather than deleted silently.
""")
    print("  README.md")

    # ---- 5. manifest: counts and checksums so the release is verifiable -----------------------
    import hashlib
    man = {}
    for f in sorted(REL.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.json":
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            n = sum(1 for l in f.read_text().splitlines() if l.strip()) if f.suffix == ".jsonl" else None
            man[str(f.relative_to(REL))] = dict(sha256=h, bytes=f.stat().st_size, items=n)
    (REL / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    print("  MANIFEST.json")
    print(f"\nrelease at {REL}\nbackup at {BAK}")


if __name__ == "__main__":
    main()
