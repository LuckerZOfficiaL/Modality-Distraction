r"""Build the public MoGround release: the tree that gets uploaded to Hugging Face.

Notes:

  * paper names        moground_base / moground_human / moground_retrieved
                       (107 shipped "moground" / "assembled", which no longer match the paper)
  * two annotators     both raters' verdicts ship for both audited pools, with `reason`.
                       107 dropped `reason` on the grounds that no claim used it; the paper now
                       quotes the rejection breakdown (10 mis-keys / 4 intended-fails / 2 leaks),
                       so it is part of a claim and has to be checkable.
  * images             NOT bundled. DCI images are SA-1B (Meta research license, manual download)
                       and the retrieved pool's distractors are CC3M, neither of which may be
                       redistributed. We ship images/MANIFEST.jsonl (source, file, sha256, bytes)
                       plus prepare_images.py, which rebuilds the folders from the upstream
                       datasets and verifies every checksum.
  * dataset card       README.md carries HF YAML frontmatter (license, configs, task categories)
                       and a per-source license table.

    python scripts/build_public_release.py            # -> data/public_release/
    python scripts/build_public_release.py --check    # verify an existing tree, write nothing
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
OUT = DR / "public_release"

# image directory -> (source key, upstream dataset, how prepare_images.py fetches it)
IMAGE_SOURCES = {
    "data/raw/dci/photos": "dci",
    "data/raw/vistext/images": "vistext",
    "data/dm/roco/images": "roco",
    "data/dm/semart/images": "semart",
    "data/pretraining/cc3m_subset/images": "cc3m",
    "data/raw/v_mcq/aokvqa": "aokvqa",
}

AUDIT = {
    "base": [("data/moground_validation/validations.jsonl", 1, "yes"),
             ("audit/verdicts_moground.jsonl", 2, "no_leak")],
    "retrieved": [("data/assembled_validation/validations.jsonl", 1, "irrelevant"),
                  ("audit/verdicts_assembled.jsonl", 2, "irrelevant")],
}


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def opts(o):
    return ast.literal_eval(o) if isinstance(o, str) else o


def src_of(image_path: str) -> str:
    s = str(image_path)
    for d, key in IMAGE_SOURCES.items():
        if d in s:
            return key
    raise SystemExit(f"unmapped image directory: {s}")


def rel_img(image_path: str) -> str:
    return f"images/{src_of(image_path)}/{Path(image_path).name}"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"  {path.relative_to(OUT)}  ({len(rows)} items)")


_SEEN_IDS: dict[str, int] = {}


def item(r, domain):
    """`uid` is the primary key. `candidate_id` joins to the paper's artifacts but is NOT unique:
    13 MoGround-Base ids are reused by a second question written on the same image."""
    cid = r["candidate_id"]
    _SEEN_IDS[cid] = _SEEN_IDS.get(cid, 0) + 1
    n = _SEEN_IDS[cid]
    return dict(uid=cid if n == 1 else f"{cid}#{n}",
                candidate_id=cid, seed_id=r.get("seed_id"), domain=domain,
                label=r["label"], image=rel_img(r["image_path"]),
                caption=r.get("caption_for_filter") or "",
                original_caption=r.get("original_caption") or "",
                question=r["question"], options=opts(r["options"]),
                correct_index=int(r["correct_index"]), rationale=r.get("rationale") or "")


def audit_rows(task):
    """Both annotators, one record each, last verdict per item wins (re-judgements overwrite)."""
    out = []
    for rel, who, passing in AUDIT[task]:
        latest = {}
        for r in jl(ROOT / rel):
            latest[r["candidate_id"]] = r
        for cid, r in latest.items():
            out.append(dict(candidate_id=cid, annotator=who, verdict=r["verdict"],
                            passes=r["verdict"] == passing, reason=r.get("reason") or "",
                            domain=r.get("domain"), modality=r.get("modality"),
                            pool=r.get("pool"), seed=r.get("seed")))
    return out


def main(check_only=False):
    if OUT.exists() and not check_only:
        shutil.rmtree(OUT)

    # ---- 1. MoGround-Base ----------------------------------------------------------------
    print("building:")
    allb = []
    for sp in ("train", "val", "test"):
        rows = []
        for r in jl(DR / f"dm/multidomain_v1/splits/{sp}.jsonl"):
            d = item(r, r.get("source"))
            d.update(split=sp, pass_v=r.get("pass_v"), pass_t=r.get("pass_t"),
                     pass_vt=r.get("pass_vt"))
            rows.append(d)
        write_jsonl(OUT / f"moground_base/splits/{sp}.jsonl", rows)
        allb += rows
    write_jsonl(OUT / "moground_base/items.jsonl", allb)
    write_jsonl(OUT / "moground_base/human_audit.jsonl", audit_rows("base"))

    nb = DR / "moground_validation/validate.ipynb"
    if nb.exists():                      # ship the interface, not one session's outputs
        j = json.loads(nb.read_text())
        for cell in j.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"], cell["execution_count"] = [], None
        (OUT / "moground_base/human_audit_notebook.ipynb").write_text(json.dumps(j, indent=1))
        print("  moground_base/human_audit_notebook.ipynb")

    _SEEN_IDS.clear()
    # ---- 2. MoGround-Human ---------------------------------------------------------------
    hrows = []
    for r in jl(DR / "dm/human_moground/frozen_20260801.jsonl"):
        d = item(r, r.get("source"))
        for a, b in (("overlap_before", "distractor_overlap_source"),
                     ("overlap_after", "distractor_overlap_authored")):
            if r.get(a) not in (None, ""):
                d[b] = round(float(r[a]), 4)
        hrows.append(d)
    write_jsonl(OUT / "moground_human/items.jsonl", hrows)
    shutil.copy2(DR / "dm/human_moground/conflict_audit.json",
                 OUT / "moground_human/caption_audit.json")
    print("  moground_human/caption_audit.json")

    _SEEN_IDS.clear()
    # ---- 3. MoGround-Retrieved (post-audit) ----------------------------------------------
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
        d = item(r, "aokvqa" if r["label"] == "vision" else "race_high")
        # here the "original" caption belongs to the retrieved distractor image, not to the item
        d["distractor_image_caption"] = d.pop("original_caption")
        d.update(half=half.get(cid), source_benchmark=r.get("source_text_qa") or "aokvqa",
                 retrieval_score=(round(float(r["retrieval_score"]), 4)
                                  if r.get("retrieval_score") not in (None, "") else None))
        arows.append(d)
    write_jsonl(OUT / "moground_retrieved/items.jsonl", arows)
    write_jsonl(OUT / "moground_retrieved/human_audit.jsonl", audit_rows("retrieved"))
    (OUT / "moground_retrieved/caption_irrelevance_audit.json").write_text(json.dumps(
        {"audited": cert["n"], "verdict_counts": cert["counts"],
         "dropped_candidate_ids": sorted(viol),
         "note": "Vision items only. The retrieved caption was judged for whether it asserts or "
                 "contradicts an option. Items listed here are excluded from items.jsonl."},
        indent=2))
    print("  moground_retrieved/caption_irrelevance_audit.json")

    # ---- 4. image manifest ---------------------------------------------------------------
    used = {}
    for pool, rows in (("moground_base", allb), ("moground_human", hrows),
                       ("moground_retrieved", arows)):
        for r in rows:
            used.setdefault(r["image"], set()).add(pool)
    man = []
    for relp in sorted(used):
        source, name = relp.split("/")[1], relp.split("/")[2]
        orig = next(k for k, v in IMAGE_SOURCES.items() if v == source)
        f = ROOT / orig / name
        man.append(dict(file=relp, source=source, upstream_id=Path(name).stem,
                        sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                        bytes=f.stat().st_size, used_in=sorted(used[relp])))
    write_jsonl(OUT / "images/MANIFEST.jsonl", man)

    # ---- 5. fetch script, card, license, checksums ---------------------------------------
    (OUT / "prepare_images.py").write_text(PREPARE_PY)
    print("  prepare_images.py")
    nv = sum(r["label"] == "vision" for r in arows)
    per_src = {}
    for m in man:
        per_src[m["source"]] = per_src.get(m["source"], 0) + 1
    (OUT / "README.md").write_text(card(allb, hrows, arows, nv, man, per_src))
    print("  README.md")
    (OUT / "LICENSE").write_text(LICENSE_TXT)
    print("  LICENSE")

    files = {}
    for f in sorted(OUT.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.json":
            files[str(f.relative_to(OUT))] = dict(
                sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size,
                items=(sum(1 for l in f.read_text().splitlines() if l.strip())
                       if f.suffix == ".jsonl" else None))
    (OUT / "MANIFEST.json").write_text(json.dumps(files, indent=2))
    print("  MANIFEST.json")

    print(f"\n{len(allb)} base / {len(hrows)} human / {len(arows)} retrieved "
          f"({nv} vision, {len(arows) - nv} text), {len(man)} unique images "
          f"({sum(m['bytes'] for m in man) / 1e6:.0f} MB, not shipped)")
    print(f"tree at {OUT}")


PREPARE_PY = r'''#!/usr/bin/env python
"""Rebuild the MoGround image folders from their upstream datasets.

No image bytes ship with this dataset: the photo sources (SA-1B via DCI, and CC3M) do not permit
redistribution. This script fetches each image from its original home and verifies it against the
sha256 in images/MANIFEST.jsonl, so the result is byte-identical to what the paper was run on.

    python prepare_images.py --sources aokvqa cc3m            # fully automatic (Hugging Face)
    python prepare_images.py --sources dci --from /path/to/sa_000138   # local, see below

Automatic:
  aokvqa   HuggingFaceM4/A-OKVQA           images keyed by question_id
  cc3m     pixparse/cc3m-wds (streaming)   images keyed by shard key; a full pass takes a while

Manual (the licence requires you to obtain these yourself, then point --from at the folder):
  dci      SA-1B shard sa_000138.tar, after accepting Meta's licence at
           https://ai.meta.com/datasets/segment-anything-downloads/
  vistext  https://vis.csail.mit.edu/vistext/  (images.zip)
  semart   SemArt (Garcia & Vogiatzis 2018), images from the Web Gallery of Art
  roco     ROCOv2, the radiology subset of PMC Open Access

Verify what you already have:

    python prepare_images.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAN = [json.loads(l) for l in (HERE / "images/MANIFEST.jsonl").read_text().splitlines() if l.strip()]
AUTO = {"aokvqa", "cc3m"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(source):
    out = []
    for m in MAN:
        if m["source"] != source:
            continue
        dst = HERE / m["file"]
        if dst.exists() and sha(dst) == m["sha256"]:
            continue
        out.append(m)
    return out


def from_local(source, root):
    todo = need(source)
    root = Path(root)
    ok = bad = 0
    for m in todo:
        name = Path(m["file"]).name
        hits = list(root.rglob(name))
        if not hits:
            continue
        dst = HERE / m["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hits[0], dst)
        if sha(dst) == m["sha256"]:
            ok += 1
        else:
            bad += 1
            print(f"  CHECKSUM MISMATCH {name}")
    print(f"{source}: copied {ok}, mismatched {bad}, still missing {len(need(source))}")


def from_aokvqa():
    from datasets import load_dataset
    todo = {Path(m["file"]).stem: m for m in need("aokvqa")}
    if not todo:
        print("aokvqa: complete"); return
    print(f"aokvqa: fetching {len(todo)}")
    for split in ("train", "validation", "test"):
        if not todo:
            break
        for ex in load_dataset("HuggingFaceM4/A-OKVQA", split=split):
            m = todo.pop(ex["question_id"], None)
            if m is None:
                continue
            dst = HERE / m["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            ex["image"].convert("RGB").save(dst, "JPEG", quality=95)
    print(f"aokvqa: {len(need('aokvqa'))} still missing")


def from_cc3m():
    from datasets import load_dataset
    todo = {Path(m["file"]).stem: m for m in need("cc3m")}
    if not todo:
        print("cc3m: complete"); return
    print(f"cc3m: streaming cc3m-wds for {len(todo)} keys (slow, one full pass)")
    for ex in load_dataset("pixparse/cc3m-wds", split="train", streaming=True):
        m = todo.pop(ex.get("__key__"), None)
        if m is None:
            continue
        dst = HERE / m["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        ex["jpg"].convert("RGB").save(dst, "JPEG", quality=95)
        if not todo:
            break
    print(f"cc3m: {len(need('cc3m'))} still missing")


def verify():
    bad = miss = 0
    for m in MAN:
        p = HERE / m["file"]
        if not p.exists():
            miss += 1
        elif sha(p) != m["sha256"]:
            bad += 1
            print(f"  MISMATCH {m['file']}")
    print(f"{len(MAN)} images: {len(MAN) - miss - bad} ok, {miss} missing, {bad} corrupt")
    return miss == bad == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--from", dest="src_dir", default=None,
                    help="local folder to search, for the manually obtained sources")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        raise SystemExit(0 if verify() else 1)
    for s in a.sources:
        if s in AUTO and not a.src_dir:
            {"aokvqa": from_aokvqa, "cc3m": from_cc3m}[s]()
        elif a.src_dir:
            from_local(s, a.src_dir)
        else:
            print(f"{s}: needs --from /path/to/local/copy (see the module docstring)")
'''


LICENSE_TXT = """MoGround annotations are released under Creative Commons Attribution-NonCommercial
4.0 International (CC BY-NC 4.0): https://creativecommons.org/licenses/by-nc/4.0/

This covers what we contribute: the questions, answer options, grounding labels, the
three-condition certificates, the splits, the audit verdicts and the retrieval scores.

Fields derived from upstream datasets keep their own terms, and no image files are redistributed
here. See the per-source table in README.md before using this data. In particular the DCI captions
are CC BY-NC 4.0, the RACE passages are for non-commercial research use, and the SA-1B and CC3M
images must be obtained from their original distributors under their own licences.
"""


def card(allb, hrows, arows, nv, man, per_src):
    nb = len(allb)
    return f"""---
license: cc-by-nc-4.0
task_categories:
  - visual-question-answering
  - multiple-choice
language:
  - en
tags:
  - vision-language
  - robustness
  - modality-distraction
  - evaluation
size_categories:
  - 1K<n<10K
configs:
  - config_name: base
    data_files:
      - split: train
        path: moground_base/splits/train.jsonl
      - split: validation
        path: moground_base/splits/val.jsonl
      - split: test
        path: moground_base/splits/test.jsonl
  - config_name: human
    data_files:
      - split: test
        path: moground_human/items.jsonl
  - config_name: retrieved
    data_files:
      - split: test
        path: moground_retrieved/items.jsonl
---

# MoGround

Three resources for measuring **modality distraction**: a model answers a question correctly from
one modality alone, then answers it wrongly once answer-irrelevant context arrives in the other
modality.

Every item is checked to be answerable from exactly one modality, so the distraction it measures
cannot be explained by the question being unanswerable.

| pool | items | what it is |
|---|---|---|
| `moground_base` | {nb} | oracle-certified single-modality items over four visual domains, with a fixed 60/20/20 split |
| `moground_human` | {len(hrows)} | hand-authored hard subset, captions written to tempt a specific wrong option |
| `moground_retrieved` | {len(arows)} | companion pool ({nv} A-OKVQA vision / {len(arows) - nv} RACE-high text), cross-modal distractors retrieved by cosine similarity, post-audit |

## Files

| path | what it holds |
|---|---|
| `moground_base/items.jsonl`, `moground_base/splits/*.jsonl` | the items, whole and split |
| `moground_base/human_audit.jsonl` | 250 sampled items judged by two annotators, with the reason for each rejection |
| `moground_base/human_audit_notebook.ipynb` | the annotation interface; the seed inside reproduces the sample and its order |
| `moground_human/items.jsonl`, `moground_human/caption_audit.json` | the hard subset and the per-caption judgement of whether it asserts an option |
| `moground_retrieved/items.jsonl` | the post-audit pool, the one every published number uses |
| `moground_retrieved/caption_irrelevance_audit.json` | verdict counts and the candidate ids the audit dropped |
| `moground_retrieved/human_audit.jsonl` | 250 sampled items judged by two annotators |
| `images/MANIFEST.jsonl` | one record per image: source, upstream id, sha256, size |
| `prepare_images.py` | rebuilds the image folders from the upstream datasets and verifies every checksum |

## Item fields

`uid`, `candidate_id`, `seed_id`, `domain`, `label` (`vision` / `text`), `image`, `caption`,
`original_caption`, `question`, `options`, `correct_index`, `rationale`.

`uid` is the primary key. `candidate_id` is what the paper's artifacts join on, and it is not
unique: 13 base items share an id with a second question written on the same image, and those
carry `uid` of the form `<candidate_id>#2`.

* **base** adds `split` and the three-condition certificate `pass_v` / `pass_t` / `pass_vt`: the
  oracle's correctness with the image alone, the caption alone, and both. A vision-grounded item
  satisfies `pass_v and pass_vt and not pass_t`, a text-grounded item the mirror of that.
* **human** adds `distractor_overlap_source` and `distractor_overlap_authored`, the lexical overlap
  between the wrong option a caption targets and, respectively, the source caption and the
  authored one.
* **retrieved** adds `half` (`dev` / `heldout`), `source_benchmark`, `retrieval_score`, and
  `distractor_image_caption` in place of `original_caption`, since there the retrieved image's own
  caption describes the distractor rather than the item.

## Images

Image files are **not** included. Two of the six sources forbid redistribution, so shipping the
rest would leave a half-usable dataset with an unclear licence. Instead every image is listed in
`images/MANIFEST.jsonl` with its upstream id and sha256, and `prepare_images.py` rebuilds the
folders:

```bash
python prepare_images.py --sources aokvqa cc3m          # automatic, from Hugging Face
python prepare_images.py --sources dci --from /path/to/extracted/sa_000138
python prepare_images.py --verify                       # checks all {len(man)} checksums
```

| source | images | where images come from | terms |
|---|---|---|---|
| `dci` | {per_src.get('dci', 0)} | SA-1B shard `sa_000138.tar`, downloaded manually from Meta | SA-1B research licence; DCI captions CC BY-NC 4.0 |
| `vistext` | {per_src.get('vistext', 0)} | VisText (`vis.csail.mit.edu/vistext`) | VisText terms |
| `semart` | {per_src.get('semart', 0)} | SemArt, images from the Web Gallery of Art | non-commercial / educational use |
| `roco` | {per_src.get('roco', 0)} | ROCOv2, PMC Open Access radiology | per-article CC licences |
| `cc3m` | {per_src.get('cc3m', 0)} | `pixparse/cc3m-wds` | Google CC3M terms, images not redistributable |
| `aokvqa` | {per_src.get('aokvqa', 0)} | `HuggingFaceM4/A-OKVQA` (COCO images) | COCO CC BY 4.0, A-OKVQA Apache-2.0 |

Text passages in the retrieved pool come from RACE-high, which is for non-commercial research use.

## How to score distraction

Run each item three times: image only (V), context only (T), and both (V+T). Condition on the items
a model answers correctly from its grounding modality alone, then

```
v-distraction = P(V+T wrong | V-only correct)   over vision-grounded items
t-distraction = P(V+T wrong | T-only correct)   over text-grounded items
```

The conditioning is model-relative on purpose. Each model is scored on the items it itself solves
from the grounded modality, so distraction always measures a capability the model demonstrably has
and then loses.

## Notes

* Certification is operational. Single-modality answerability holds relative to the oracles used,
  not as absolute ground truth. A cross-oracle check on a stratified 240-item sample puts
  independent-oracle agreement at 94%.
* The retrieved pool ships post-audit. The dropped items are listed rather than deleted silently.
* `MANIFEST.json` gives a sha256 and an item count for every file in this repo.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    main(ap.parse_args().check)
