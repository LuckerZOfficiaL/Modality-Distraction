r"""Score proprietary models on \dsname-Human under the three-condition protocol.

Why the human set: it carries no oracle certificate, so evaluating models from the same families as
the oracles that certified \dsname{} introduces no circularity. It is also the hardest surface we
have (base v-distraction $0.274$ vs $0.076$ on \dsname{} test).

Prompt parity with the open-model runs is the point of this script. `sae_steering.steering.
build_messages` produces, in order: an optional `Caption: ...` text block, the image, then
`Question: ...` with the lettered options and `Reply with exactly one letter (A, B, C, or D) and
nothing else.` We reproduce that ordering exactly, in a single user turn with no system message,
and drop the caption block (T-absent) or the image (V-absent) per condition.

Extraction differs from the open models, which were scored by argmax over the four letter logits.
Closed APIs expose no logits, so:
  gemini  enum-constrained decoding (responseSchema over {A,B,C,D}) makes it a forced choice, so
          there are no unparseable outputs by construction: the nearest available analogue of argmax
  claude  free generation parsed for a single letter; unparseable -> -1, scored wrong, and the rate
          is reported (subagents write the answers, see --provider claude-prep / claude-collect)

    python scripts/108_proprietary_human_eval.py --provider gemini --model gemini-3.5-flash
    python scripts/108_proprietary_human_eval.py --provider claude-prep
    python scripts/108_proprietary_human_eval.py --provider claude-collect --key hm3_sonnet5
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
ITEMS = DR / "dm/human_moground/frozen_20260801.jsonl"
COND = [("VT", True, True), ("V", True, False), ("T", False, True)]   # (name, use_image, use_caption)
LETTER = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
MAX_DIM = 1024


def rows():
    out = []
    for line in ITEMS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        o = r["options"]
        r["options"] = ast.literal_eval(o) if isinstance(o, str) else o
        r["correct_index"] = int(r["correct_index"])
        out.append(r)
    return out


def prompt_parts(r, use_image, use_caption):
    """The canonical prompt of build_messages, as ordered (kind, value) parts."""
    opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", r["options"]))
    parts = []
    cap = r.get("caption_for_filter")
    if use_caption and cap:
        parts.append(("text", f"Caption: {cap}"))
    if use_image:
        parts.append(("image", r["image_path"]))
    parts.append(("text", f"Question: {r['question']}\n{opts}\n\n"
                          "Reply with exactly one letter (A, B, C, or D) and nothing else."))
    return parts


def parse(txt):
    m = LETTER.search((txt or "").strip())
    return ord(m.group(1).upper()) - ord("A") if m else -1


def write_eval(key, recs):
    out = DR / f"eval/t8/{key}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    n_bad = sum(1 for r in recs for k in ("pred_vt", "pred_v", "pred_t") if r[k] < 0)
    print(f"wrote {out}  ({len(recs)} items, {n_bad} unparseable of {3*len(recs)} answers)")
    v = [r for r in recs if r["label"] == "vision" and r["pred_v"] == r["correct_index"]]
    if v:
        d = sum(r["pred_vt"] != r["correct_index"] for r in v)
        print(f"  v-distraction {d}/{len(v)} = {d/len(v):.3f}")
    t = [r for r in recs if r["label"] == "text" and r["pred_t"] == r["correct_index"]]
    if t:
        d = sum(r["pred_vt"] != r["correct_index"] for r in t)
        print(f"  t-distraction {d}/{len(t)} = {d/len(t):.3f}")


def run_gemini_sync(model, key, workers=8):
    """Direct REST calls with enum-constrained decoding, run concurrently.

    Batch would be 50% cheaper but the installed google-genai rejects the Files-API source, and at
    375 requests the discount is negligible against a multi-hour polling window. For the much larger
    assembled-pool run the batch path is worth fixing.
    """
    import base64, io, urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    kp = Path(".credentials/google_gladia")
    api = kp.read_text().strip().split()[0]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api}"

    def img_part(path):
        im = Image.open(path).convert("RGB")
        im.thumbnail((MAX_DIM, MAX_DIM))
        b = io.BytesIO(); im.save(b, "JPEG", quality=90)
        return {"inline_data": {"mime_type": "image/jpeg",
                                "data": base64.b64encode(b.getvalue()).decode()}}

    def one(job):
        r, cname, ui, uc = job
        parts = [img_part(v) if k == "image" else {"text": v}
                 for k, v in prompt_parts(r, ui, uc)]
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048,
                                     "responseMimeType": "text/x.enum",
                                     "responseSchema": {"type": "STRING",
                                                        "enum": ["A", "B", "C", "D"]}}}
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    d = json.load(resp)
                return (r["candidate_id"], cname,
                        d["candidates"][0]["content"]["parts"][0]["text"])
            except Exception:
                if attempt == 4:
                    return (r["candidate_id"], cname, None)
                import time as _t; _t.sleep(2 * (attempt + 1))

    rs = rows()
    jobs = [(r, c, ui, uc) for r in rs for c, ui, uc in COND]
    print(f"{len(jobs)} requests ({len(rs)} items x 3 conditions) -> {model}, {workers} workers")
    got = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (cid, cname, txt) in enumerate(ex.map(one, jobs), 1):
            got[(cid, cname)] = txt
            if i % 75 == 0:
                print(f"   {i}/{len(jobs)}")
    recs = []
    for r in rs:
        preds = [parse(got.get((r["candidate_id"], c))) for c, _, _ in COND]
        recs.append(dict(candidate_id=r["candidate_id"], label=r["label"], source=r.get("source"),
                         correct_index=r["correct_index"],
                         pred_vt=preds[0], pred_v=preds[1], pred_t=preds[2]))
    write_eval(key, recs)


def run_gemini(model, key, poll):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from sae_steering.oracle_batch import submit_and_wait
    from sae_steering.oracle_clients import ImageInput, make_client

    rs = rows()
    reqs = []
    for r in rs:
        for cname, ui, uc in COND:
            parts = [(k, ImageInput(path=v, max_dim=MAX_DIM) if k == "image" else v)
                     for k, v in prompt_parts(r, ui, uc)]
            reqs.append(dict(key=f"{r['candidate_id']}::{cname}", parts=parts, temperature=0.0,
                             max_output_tokens=2048,            # room for thinking before the letter
                             response_mime_type="text/x.enum",
                             response_schema={"type": "STRING", "enum": ["A", "B", "C", "D"]}))
    print(f"{len(reqs)} requests ({len(rs)} items x 3 conditions) -> {model} batch")
    kp = ROOT / ".credentials/google_gladia"
    if not kp.exists():
        kp = Path(".credentials/google_gladia")
    client = make_client("gemini", model=model, key_path=str(kp))
    res = submit_and_wait(gemini_client=client, requests=reqs,
                          work_dir=DR / f"proprietary/{key}", display_name=f"human-{key}",
                          poll_seconds=poll)
    got = {x["key"]: x for x in res}
    recs = []
    for r in rs:
        preds = []
        for cname, _, _ in COND:
            g = got.get(f"{r['candidate_id']}::{cname}", {})
            preds.append(parse(g.get("text")))
        recs.append(dict(candidate_id=r["candidate_id"], label=r["label"], source=r.get("source"),
                         correct_index=r["correct_index"],
                         pred_vt=preds[0], pred_v=preds[1], pred_t=preds[2]))
    write_eval(key, recs)


def claude_prep(chunk):
    """Write per-condition chunk files for subagents to answer."""
    d = DR / "proprietary/claude_human"
    for sub in ("chunks", "answers"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    rs = rows()
    made = []
    for cname, ui, uc in COND:
        items = []
        for r in rs:
            rec = dict(candidate_id=r["candidate_id"], question=r["question"], options=r["options"])
            if uc and r.get("caption_for_filter"):
                rec["caption"] = r["caption_for_filter"]
            if ui:
                rec["image_path"] = r["image_path"]
            items.append(rec)
        size = chunk if ui else len(items)          # text-only needs no image reads: one chunk
        for i in range(0, len(items), size):
            p = d / f"chunks/{cname}_{i//size}.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in items[i:i + size]) + "\n")
            made.append(p)
    print(f"wrote {len(made)} chunk files under {d/'chunks'}")
    for p in made:
        print("   ", p.name, sum(1 for _ in p.read_text().splitlines() if _.strip()), "items")


def claude_collect(key):
    d = DR / "proprietary/claude_human"
    got = {}
    for p in sorted((d / "answers").glob("*.jsonl")):
        cname = p.stem.split("_")[0]
        for line in p.read_text().splitlines():
            if line.strip():
                a = json.loads(line)
                got[(a["candidate_id"], cname)] = a.get("answer", a.get("predicted_index"))
    rs = rows()
    recs, miss = [], 0
    for r in rs:
        preds = []
        for cname, _, _ in COND:
            v = got.get((r["candidate_id"], cname))
            if v is None:
                miss += 1; preds.append(-1)
            elif isinstance(v, int):
                preds.append(v)
            else:
                preds.append(parse(str(v)))
        recs.append(dict(candidate_id=r["candidate_id"], label=r["label"], source=r.get("source"),
                         correct_index=r["correct_index"],
                         pred_vt=preds[0], pred_v=preds[1], pred_t=preds[2]))
    if miss:
        print(f"WARNING: {miss} missing answers (scored -1)")
    write_eval(key, recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True,
                    choices=["gemini", "claude-prep", "claude-collect"])
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--key", default=None, help="output key, default hm3_<model>")
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--sync", action="store_true", help="concurrent sync instead of batch")
    a = ap.parse_args()
    if a.provider == "gemini":
        (run_gemini_sync(a.model, a.key or f"hm3_{a.model.replace('.','').replace('-','')}_w0.0")
         if a.sync else run_gemini(a.model, a.key or f"hm3_{a.model.replace('.', '').replace('-', '')}_w0.0", a.poll))
    elif a.provider == "claude-prep":
        claude_prep(a.chunk)
    else:
        claude_collect(a.key or "hm3_sonnet5_w0.0")


if __name__ == "__main__":
    main()
