"""Adversarial captions for Method-A (hard-negative / routing) distraction-aware finetuning.

For each VISION-grounded train item we synthesise an `adv_caption`: a fluent, on-topic caption of
the SAME scene that asserts the detail of a specific WRONG option, so a model that trusts the caption
would flip to the distractor. Finetuning on these teaches ROUTE-ON-RELEVANCE (override the caption
when the image contradicts it) rather than the naive "downweight text globally" that Phase-0 dropout
learned (which cost -4pp on legitimate caption-use). Text-grounded items are left untouched -- they
carry the preservation signal.

Backend: Gemini Batch API (50% discount, file-based; the user runs Gemini). The image is included so
the fabricated caption stays scene-consistent and therefore adversarial rather than nonsensical.

    # 1) generate (user, GPU-free, Gemini API):
    python scripts/76_adversarial_captions.py --generate
    # 2) merge into an augmented train split (CPU):
    python scripts/76_adversarial_captions.py --merge
    # 3) finetune Method A:
    CUDA_VISIBLE_DEVICES=0 python scripts/75_finetune_distraction.py --method hardneg \
        --train data/dm/multidomain_v1/splits/train_adv.jsonl --out runs/ft_hardneg_qwen
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

LETTERS = ["A", "B", "C", "D"]

SYSTEM = (
    "You write image captions for a research dataset. Given an image, a multiple-choice question "
    "about it, the options, and which option is CORRECT, you fabricate a fluent, natural-sounding "
    "caption of the SAME scene that would mislead a careful reader into choosing a specified WRONG "
    "option. The caption must stay consistent with everything else in the scene and only bend the "
    "single detail the question is about. Never hedge, never say the caption is wrong, and never use "
    "self-referential words like 'image', 'photo', 'caption', 'shown', or 'according to'."
)

USER_TMPL = (
    "QUESTION: {q}\n"
    "OPTIONS:\n{opts}\n"
    "CORRECT OPTION: {gold_letter}. {gold_text}\n\n"
    "TRUE CAPTION (accurate; for scene grounding only):\n{cap}\n\n"
    "Task: pick the WRONG option that is most plausibly confusable with the correct one, then write "
    "ONE adversarial caption (roughly {nchar} characters, similar in style and length to the true "
    "caption) that describes this scene as if that wrong option were true. Keep every other detail "
    "faithful; only the detail the question asks about should point to the wrong option.\n"
    'Return JSON: {{"target_index": <int 0-based of the wrong option you argued for>, '
    '"adv_caption": "<the caption>"}}'
)

FORBIDDEN_RE = re.compile(
    r"\b(image|picture|photo|photograph|caption|shown|depicted|pictured|displayed|according to)\b",
    re.IGNORECASE)


def _rows(cfg, split="train"):
    dm = Path(cfg["paths"]["dm_dir"])
    return [json.loads(l) for l in (dm / f"splits/{split}.jsonl").read_text().splitlines() if l.strip()]


def _work_dir(cfg, split, suffix=""):
    # keep the original train paths byte-identical; suffix only for other splits / other arms
    dm = Path(cfg["paths"]["dm_dir"])
    base = "adv_captions" if split == "train" else f"adv_captions_{split}"
    return dm / (base + suffix)


def _strip_json(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s); s = re.sub(r"\n?```$", "", s)
    return s.strip()


def cmd_generate(cfg, args):
    from moground.oracle_clients import GeminiClient, ImageInput
    from moground.oracle_batch import submit_and_wait

    work = _work_dir(cfg, args.split, args.work_suffix); work.mkdir(parents=True, exist_ok=True)
    # --labels all: ignore the grounding certificate and generate for EVERY item. This is the
    # certificate-free arm: a practitioner without per-item grounding labels cannot know which
    # items are vision-grounded, so they would fabricate a misleading caption for all of them.
    vision = [r for r in _rows(cfg, args.split)
              if args.labels == "all" or r["label"] == "vision"]
    if args.limit:
        vision = vision[:args.limit]

    # submit_and_wait caches its job in work/work/ and RESUMES whenever batch.txt exists
    # (rebuilding the request file only when it is absent). A leftover cache from a different
    # request count (e.g. an earlier --limit smoke) would otherwise be resumed silently and
    # return "no response" for the extra inputs. Invalidate a stale/short cache here.
    wd = work / "work"
    in_path = wd / "batch_input.jsonl"
    if args.fresh or (in_path.exists()
                      and sum(1 for _ in in_path.open()) != len(vision)):
        for name in ("batch_input.jsonl", "batch.txt", "batch_output.jsonl"):
            (wd / name).unlink(missing_ok=True)
        print(f"cleared stale batch cache in {wd}")

    reqs = []
    for r in vision:
        opts = r["options"]; gi = int(r["correct_index"])
        cap = r.get("caption_for_filter") or r.get("original_caption") or ""
        user = USER_TMPL.format(
            q=r["question"],
            opts="\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)),
            gold_letter=LETTERS[gi], gold_text=opts[gi], cap=cap, nchar=max(len(cap), 120))
        reqs.append({"key": r["candidate_id"], "system": SYSTEM, "user": user,
                     "images": [ImageInput(path=r["image_path"], max_dim=768)],
                     "temperature": 0.7, "response_mime_type": "application/json",
                     "max_output_tokens": 4096})

    client = GeminiClient(model=args.model, key_path=args.key_path)
    print(f"submitting {len(reqs)} adversarial-caption requests ({args.split} split) via Gemini Batch API...")
    out = submit_and_wait(gemini_client=client, requests=reqs, work_dir=work / "work",
                          display_name=f"adv-captions-{args.split}-{args.model}", poll_seconds=args.poll)

    n_ok = 0
    with (work / "adv_captions.jsonl").open("w") as f:
        for item in out:
            rec = {"candidate_id": item["key"], "adv_caption": None, "target_index": None,
                   "error": item.get("error")}
            if item.get("text"):
                try:
                    d = json.loads(_strip_json(item["text"]))
                    ac = (d.get("adv_caption") or "").strip()
                    if ac:
                        rec["adv_caption"] = ac; rec["target_index"] = d.get("target_index"); n_ok += 1
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"parse: {e}"
            f.write(json.dumps(rec) + "\n")
    print(f"-> {work/'adv_captions.jsonl'}  ({n_ok}/{len(reqs)} usable captions)")


def cmd_merge(cfg, args):
    dm = Path(cfg["paths"]["dm_dir"])
    adv_path = _work_dir(cfg, args.split, args.work_suffix) / "adv_captions.jsonl"
    adv = {}
    forbidden = 0
    for l in adv_path.read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        c = d.get("adv_caption")
        if c and not FORBIDDEN_RE.search(c):
            adv[d["candidate_id"]] = (c, d.get("target_index"))
        elif c:
            forbidden += 1
    rows = _rows(cfg, args.split); n = 0
    for r in rows:
        if (args.labels == "all" or r["label"] == "vision") and r["candidate_id"] in adv:
            r["adv_caption"], r["adv_target_index"] = adv[r["candidate_id"]]; n += 1
    out = dm / "splits" / (f"{args.split}_adv.jsonl" if not args.out_suffix
                           else f"{args.split}_adv{args.out_suffix}.jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nV = len(rows) if args.labels == "all" else sum(x["label"] == "vision" for x in rows)
    print(f"adv captions merged: {n}/{nV} items have adv_caption "
          f"({forbidden} dropped for self-referential words)")
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--key-path", default=".credentials/google_gladia")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--fresh", action="store_true", help="wipe any cached batch job before submitting")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="which D_M split to generate/merge adversarial captions for")
    ap.add_argument("--limit", type=int, default=0, help="0 = all vision train items (smoke: e.g. 8)")
    ap.add_argument("--labels", default="vision", choices=["vision", "all"],
                    help="which items get an adversarial caption. vision: certificate-selected "
                         "(the paper's recipe). all: every item, certificate ignored.")
    ap.add_argument("--work-suffix", default="", help="suffix for the work dir, to keep arms apart")
    ap.add_argument("--out-suffix", default="", help="suffix for the merged train file, e.g. _all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.generate:
        cmd_generate(cfg, args)
    if args.merge:
        cmd_merge(cfg, args)
    if not (args.generate or args.merge):
        ap.error("pass --generate and/or --merge")


if __name__ == "__main__":
    main()
