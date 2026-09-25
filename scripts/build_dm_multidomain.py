"""Merge all D_M sources into a single multidomain D_M and produce train/val/test splits.

Sources (each contributes its dm_all.jsonl or survivors.jsonl):
  DCI (natural):      data/dm/sonnet/batch{1..5}/dm_all.jsonl
  VisText (charts):   data/dm/vistext/sonnet/batch_{1..5}/dm_all.jsonl
  SemArt (paintings): data/dm/semart/{pilot100,batch500,batch500b}/survivors.jsonl
  ROCO (radiology):   data/dm/roco/{pilot100,batch500}/survivors.jsonl

Each row gets two new fields for traceability:
    source   in {dci, vistext, semart, roco}
    subbatch e.g. "sonnet/batch1", "semart/batch500"

Output:
  data/dm/multidomain_v1/dm_all.jsonl
  data/dm/multidomain_v1/splits/{train,val,test}.jsonl  (60/20/20)
  data/dm/multidomain_v1/summary.json
"""
from __future__ import annotations
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / "data/dm"
OUT_DIR = DATA_ROOT / "multidomain_v1"
SPLITS_DIR = OUT_DIR / "splits"
SPLIT_FRACS = (0.6, 0.2, 0.2)
SEED = 0

SOURCES = {
    "dci": [
        ("sonnet/batch1", DATA_ROOT / "sonnet/batch1/dm_all.jsonl"),
        ("sonnet/batch2", DATA_ROOT / "sonnet/batch2/dm_all.jsonl"),
        ("sonnet/batch3", DATA_ROOT / "sonnet/batch3/dm_all.jsonl"),
        ("sonnet/batch4", DATA_ROOT / "sonnet/batch4/dm_all.jsonl"),
        ("sonnet/batch5", DATA_ROOT / "sonnet/batch5/dm_all.jsonl"),
    ],
    "vistext": [
        ("vistext/batch_1", DATA_ROOT / "vistext/sonnet/batch_1/dm_all.jsonl"),
        ("vistext/batch_2", DATA_ROOT / "vistext/sonnet/batch_2/dm_all.jsonl"),
        ("vistext/batch_3", DATA_ROOT / "vistext/sonnet/batch_3/dm_all.jsonl"),
        ("vistext/batch_4", DATA_ROOT / "vistext/sonnet/batch_4/dm_all.jsonl"),
        ("vistext/batch_5", DATA_ROOT / "vistext/sonnet/batch_5/dm_all.jsonl"),
    ],
    "semart": [
        ("semart/pilot100",  DATA_ROOT / "semart/pilot100/survivors.jsonl"),
        ("semart/batch500",  DATA_ROOT / "semart/batch500/survivors.jsonl"),
        ("semart/batch500b", DATA_ROOT / "semart/batch500b/survivors.jsonl"),
    ],
    "roco": [
        ("roco/pilot100", DATA_ROOT / "roco/pilot100/survivors.jsonl"),
        ("roco/batch500", DATA_ROOT / "roco/batch500/survivors.jsonl"),
    ],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    counts: dict[str, dict] = {}
    for source, batches in SOURCES.items():
        c_v = c_t = c_skip = 0
        for subbatch, path in batches:
            if not path.exists():
                print(f"  WARN: {path} not found, skipping")
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["source"] = source
                r["subbatch"] = subbatch
                if r["label"] == "vision":
                    c_v += 1
                elif r["label"] == "text":
                    c_t += 1
                else:
                    c_skip += 1
                    continue
                all_rows.append(r)
        counts[source] = {"V": c_v, "T": c_t, "skipped": c_skip,
                          "total": c_v + c_t}
        print(f"  {source}: V={c_v}  T={c_t}  total={c_v+c_t}")

    rng = random.Random(SEED)
    rng.shuffle(all_rows)
    n = len(all_rows)
    n_train = int(n * SPLIT_FRACS[0])
    n_val = int(n * SPLIT_FRACS[1])
    splits = {
        "train": all_rows[:n_train],
        "val":   all_rows[n_train : n_train + n_val],
        "test":  all_rows[n_train + n_val :],
    }

    # write dm_all and per-split files
    with (OUT_DIR / "dm_all.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    for name, rows in splits.items():
        with (SPLITS_DIR / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # per-split, per-source, per-label summary
    summary: dict = {"seed": SEED, "split_fracs": SPLIT_FRACS,
                     "n_total": n, "per_source": counts, "per_split": {}}
    for name, rows in splits.items():
        c = Counter((r["source"], r["label"]) for r in rows)
        per_src: dict = {}
        for src in SOURCES:
            per_src[src] = {"V": c[(src, "vision")], "T": c[(src, "text")]}
        summary["per_split"][name] = {"n": len(rows), "per_source": per_src}

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntotal: {n} rows -> {OUT_DIR / 'dm_all.jsonl'}")
    print(f"  train: {len(splits['train'])} | val: {len(splits['val'])} | test: {len(splits['test'])}")


if __name__ == "__main__":
    main()
