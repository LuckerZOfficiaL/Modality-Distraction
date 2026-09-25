"""Step 4e: build a sentence-transformer embedding index over an image pool's
captions (CC3M by default). Output is used by 04g_retrieve_text_qa_candidates.py
to find topically-related but answer-irrelevant images for text-QA items.

CHUNKED + RESUME-SAFE. Processes the caption pool in configurable chunks
(default 50K) and writes each chunk to disk before encoding the next one.
This keeps peak memory ~200 MB regardless of pool size, vs. ~6-12 GB if
1M captions are encoded in a single call. Resuming a killed run picks up at
the first missing chunk.

Output structure (under --out-dir):
  chunks/chunk_NNNN.npy           # per-chunk embeddings (float32, normalized)
  chunks/chunk_NNNN.jsonl         # per-chunk index rows
  embeddings.npy                  # final concatenated (N, D) float32, normalized
  index.parquet (or index.jsonl)  # candidate_id, image_path, caption
  meta.json                       # model name, dim, n, source path

FAISS index is NOT built (retrieval at step 04g uses a plain numpy matmul,
which is fast enough for 1M vectors and avoids holding a second 3 GB copy
in RAM).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/pretraining/cc3m_subset/index.jsonl",
                    help="jsonl with {image_path, caption} (and optional key) per line")
    ap.add_argument("--out-dir", default="data/image_caption_index/cc3m_subset")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                    help="any Sentence-Transformers model; 768-dim is plenty")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="per-step batch inside model.encode (small inner batch)")
    ap.add_argument("--chunk-size", type=int, default=50_000,
                    help="captions per persisted chunk (controls peak memory)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in Path(args.source).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "candidate_id": r.get("key") or r.get("candidate_id") or f"row_{len(rows):08d}",
            "image_path": r["image_path"],
            "caption": r["caption"],
        })
    n_total = len(rows)
    n_chunks = (n_total + args.chunk_size - 1) // args.chunk_size
    print(f"loaded {n_total} captions  ->  {n_chunks} chunks of {args.chunk_size}")

    # Resume: skip chunks already on disk with the expected shape.
    done: set[int] = set()
    for ci in range(n_chunks):
        emb_p = chunks_dir / f"chunk_{ci:04d}.npy"
        jsl_p = chunks_dir / f"chunk_{ci:04d}.jsonl"
        if emb_p.exists() and jsl_p.exists():
            expected = min(args.chunk_size, n_total - ci * args.chunk_size)
            arr = np.load(emb_p, mmap_mode="r")
            if arr.shape[0] == expected:
                done.add(ci)
    if done:
        print(f"resume: {len(done)} / {n_chunks} chunks already done")

    todo = [ci for ci in range(n_chunks) if ci not in done]
    dim = None
    if todo:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model, device=args.device)
        dim = model.get_sentence_embedding_dimension()
        print(f"model={args.model}  dim={dim}  device={args.device}")

        for j, ci in enumerate(todo, 1):
            start = ci * args.chunk_size
            end = min(start + args.chunk_size, n_total)
            chunk_rows = rows[start:end]
            captions = [r["caption"] for r in chunk_rows]
            emb = model.encode(
                captions,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            np.save(chunks_dir / f"chunk_{ci:04d}.npy", emb)
            with (chunks_dir / f"chunk_{ci:04d}.jsonl").open("w") as f:
                for r in chunk_rows:
                    f.write(json.dumps(r) + "\n")
            print(f"  chunk {ci+1:>4}/{n_chunks}  ({end - start} captions, {emb.shape[1]}-d)  "
                  f"[{j}/{len(todo)} new]")

    if dim is None:
        # all chunks done; infer dim from one
        any_chunk = next(iter(chunks_dir.glob("chunk_*.npy")))
        dim = np.load(any_chunk, mmap_mode="r").shape[1]

    # Finalize: stitch chunks into embeddings.npy via memmap, index.parquet/jsonl,
    # meta.json. memmap keeps peak memory at ~200 MB during the concat too.
    print(f"\nfinalizing: writing embeddings.npy ({n_total} x {dim} float32)...")
    emb_path = out_dir / "embeddings.npy"
    # Use np.lib.format.open_memmap so the file has a proper .npy header.
    out_mmap = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype=np.float32, shape=(n_total, dim))
    all_rows = []
    cursor = 0
    for ci in range(n_chunks):
        arr = np.load(chunks_dir / f"chunk_{ci:04d}.npy", mmap_mode="r")
        out_mmap[cursor:cursor + arr.shape[0]] = arr[:]
        cursor += arr.shape[0]
        for line in (chunks_dir / f"chunk_{ci:04d}.jsonl").read_text().splitlines():
            if line.strip():
                all_rows.append(json.loads(line))
    out_mmap.flush()
    del out_mmap
    assert cursor == n_total, f"stitch length mismatch {cursor} vs {n_total}"
    print(f"wrote {emb_path}")

    # index.parquet (fallback to jsonl if pyarrow not installed)
    try:
        import pandas as pd
        pd.DataFrame(all_rows).to_parquet(out_dir / "index.parquet", index=False)
        print(f"wrote {out_dir / 'index.parquet'}  rows={len(all_rows)}")
    except ImportError:
        with (out_dir / "index.jsonl").open("w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        print(f"pyarrow not available; wrote {out_dir / 'index.jsonl'} instead")

    (out_dir / "meta.json").write_text(json.dumps({
        "model": args.model,
        "dim": dim,
        "n": n_total,
        "source": str(args.source),
        "chunk_size": args.chunk_size,
    }, indent=2))
    print("done.")
    print(f"  retrieval (04g) loads embeddings.npy via np.load(mmap_mode='r') so")
    print(f"  query-time RAM is bounded — no need to ever hold the full {n_total*dim*4//1024**3} GB array in memory.")


if __name__ == "__main__":
    main()
