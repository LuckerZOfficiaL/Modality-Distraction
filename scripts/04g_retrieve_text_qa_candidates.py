"""Step 4g: assemble text-QA-driven candidates by retrieving topically-related
images from the precomputed image-caption index.

Per text-QA item, find the closest image (by Sentence-Transformer cosine
similarity between caption and the QA passage/question), and emit a
candidates.jsonl row in the standard schema for the existing 04c -> 04d -> 05
pipeline.

The image is paired with the QA *passage* as `caption_for_filter` (option A
from earlier design: model gets QA passage as the textual context, image as
visual context). The image itself is from a different domain (e.g. CC3M
natural photos), so even if caption and passage are very similar, the image
cannot contain the propositional answer to the QA question.

Output rows carry `label = "text"` (T-grounded by construction; the passage is
the source of truth for the correct answer, the image is the would-be
distractor). Downstream Qwen 3-pass (scripts/04c_*) then partitions them into:
  - robust T-grounded survivors: T-only ok, V-only wrong, V+T ok
  - v-distracted (image distracts joint inference): T-only ok, V-only wrong, V+T wrong

Usage:
  python scripts/04g_retrieve_text_qa_candidates.py \\
      --qa data/raw/race_mcq.jsonl \\
      --image-index data/image_caption_index/cc3m_subset \\
      --out data/dm/text_qa/race/candidates.jsonl \\
      --source-name race
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_index(idx_dir: Path):
    meta = json.loads((idx_dir / "meta.json").read_text())
    # mmap so a 3 GB embeddings.npy doesn't get fully paged into RAM.
    emb = np.load(idx_dir / "embeddings.npy", mmap_mode="r")
    # rows: prefer parquet, fall back to jsonl
    if (idx_dir / "index.parquet").exists():
        import pandas as pd
        rows = pd.read_parquet(idx_dir / "index.parquet").to_dict("records")
    else:
        rows = [json.loads(l) for l in (idx_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == emb.shape[0], f"index/embedding length mismatch {len(rows)} vs {emb.shape[0]}"
    return meta, emb, rows


def _topk(query_vec: np.ndarray, emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    # cosine via dot (both already L2-normalized).
    scores = emb @ query_vec
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True,
                    help="jsonl with {id, question, passage, options, correct_index}")
    ap.add_argument("--image-index", required=True,
                    help="dir written by 04e_build_image_caption_index.py")
    ap.add_argument("--out", required=True, help="output candidates.jsonl")
    ap.add_argument("--source-name", required=True,
                    help="short tag for the QA source (e.g. race, hotpotqa, mmlu)")
    ap.add_argument("--mode", default="t", choices=["t", "v"],
                    help="t: T-grounded source (passage + question + options + correct; "
                         "retrieved CC3M IMAGE is the would-be distractor; passage is "
                         "caption_for_filter). v: V-grounded source (image_path + question + "
                         "options + correct; retrieved CC3M CAPTION is the would-be "
                         "distractor and becomes caption_for_filter; the source image "
                         "stays as image_path).")
    ap.add_argument("--query-field", default=None, choices=["passage", "question", "qa"],
                    help="text to embed as the retrieval query. Default: 'passage' in "
                         "t-mode, 'question' in v-mode. 'qa' concatenates question + passage.")
    ap.add_argument("--top-k", type=int, default=1,
                    help="emit a candidate from the top-k retrieved images "
                         "(if k>1 use deterministic pick via row id)")
    ap.add_argument("--st-model", default="sentence-transformers/all-mpnet-base-v2",
                    help="must match the model used at index build time")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    qa_rows = [json.loads(l) for l in Path(args.qa).read_text().splitlines() if l.strip()]
    qa_rows = [r for r in qa_rows if "error" not in r]
    print(f"loaded {len(qa_rows)} QA items from {args.qa}")

    meta, emb, pool = _load_index(Path(args.image_index))
    print(f"loaded image index: n={meta['n']} dim={meta['dim']} model={meta['model']}")
    if meta["model"] != args.st_model:
        print(f"!! warning: index was built with {meta['model']} but --st-model={args.st_model}")

    # Embed queries
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.st_model, device=args.device)

    # default query field depends on mode (v-mode has no passage)
    qfield = args.query_field or ("question" if args.mode == "v" else "passage")
    def query_of(r):
        if qfield == "question": return r["question"]
        if qfield == "qa":       return r["question"] + "\n" + (r.get("passage") or "")
        return r.get("passage") or r["question"]

    queries = [query_of(r) for r in qa_rows]
    qvecs = model.encode(queries, batch_size=64, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with out_path.open("w") as f:
        for r, qv in zip(qa_rows, qvecs):
            idxs, scores = _topk(qv, emb, max(args.top_k, 1))
            # deterministic per-item pick from top-k (stable across reruns)
            pick = int(idxs[hash(str(r["id"])) % len(idxs)]) if args.top_k > 1 else int(idxs[0])
            pool_row = pool[pick]
            cand_id = f"{args.source_name}_{r['id']}"
            if args.mode == "t":
                # Source provides text; retrieved CC3M image is the distractor.
                out_row = {
                    "image_path": pool_row["image_path"],
                    "original_caption": pool_row["caption"],
                    "label": "text",
                    "caption_for_filter": r.get("passage") or "",
                    "image_pool_caption": pool_row["caption"],
                }
            else:  # mode == "v"
                # Source provides image; retrieved CC3M caption is the distractor.
                out_row = {
                    "image_path": r["image_path"],
                    "original_caption": r.get("question", ""),
                    "label": "vision",
                    "caption_for_filter": pool_row["caption"],
                    "image_pool_caption": pool_row["caption"],
                }
            f.write(json.dumps({
                "candidate_id": cand_id,
                "seed_id": cand_id,
                **out_row,
                "question": r["question"],
                "options": r["options"],
                "correct_index": r["correct_index"],
                "rationale": "",
                "source_text_qa": args.source_name,
                "retrieval_score": float(scores[0 if args.top_k == 1 else list(idxs).index(pick)]),
                "retrieval_mode": args.mode,
            }) + "\n")
            n_out += 1
    print(f"wrote {n_out} rows -> {out_path}")
    print("next: python scripts/04c_qwen_3pass_on_candidates.py --dm-dir " + str(out_path.parent))


if __name__ == "__main__":
    main()
