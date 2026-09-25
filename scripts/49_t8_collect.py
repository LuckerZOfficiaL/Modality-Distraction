"""T8: collect each VLM's 3-condition (V-only / T-only / V+T) predictions on the FULL
D_M, so D_M can be used as a cross-model modality-robustness benchmark.

Per model writes data/eval/t8/{model}.jsonl with one row per item:
  {candidate_id, source, label, correct_index, pred_vt, pred_v, pred_t}

  - qwen          : exported from the full counterfactual cf-store (canonical forward,
                    no rerun); letter_logits argmax over conds [VT, V, T].
  - llavanext     : load_llavanext  + predict_mcq (image+caption / image / caption).

GPU (llavanext). Resumable (skips candidate_ids already written).

    python scripts/49_t8_collect.py --model qwen            # instant, CPU
    CUDA_VISIBLE_DEVICES=6 python scripts/49_t8_collect.py --model llavanext
    CUDA_VISIBLE_DEVICES=6 python scripts/49_t8_collect.py --model llavanext
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def load_dm_rows(dm: Path) -> dict:
    rows = {}
    for sp in ("train", "val", "test"):
        p = dm / "splits" / f"{sp}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); rows[r["candidate_id"]] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen", "llavanext"])
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual_full")
    ap.add_argument("--rows-jsonl", default=None,
                    help="row source (default = D_M splits). For assembled: data/dm/merged_aokvqa_racehigh/dm_all.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--qwen-infer", action="store_true",
                    help="run Qwen inference on the rows (full coverage) instead of cf-store export")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dr = Path(cfg["paths"]["data_root"]); dm = Path(cfg["paths"]["dm_dir"])
    if args.rows_jsonl:
        rowmap = {json.loads(l)["candidate_id"]: json.loads(l)
                  for l in Path(args.rows_jsonl).read_text().splitlines() if l.strip()}
    else:
        rowmap = load_dm_rows(dm)
    out = Path(args.out or f"data/eval/t8/{args.model}.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "qwen" and args.qwen_infer:
        # run Qwen inference on all rows, reusing 06c's exact V/T/VT message builder
        import importlib.util, torch
        from moground.models import load_qwen_vl
        from moground.steering import get_letter_token_ids
        from qwen_vl_utils import process_vision_info
        _sp = importlib.util.spec_from_file_location("_c6", Path(__file__).parent / "06c_collect_dm_counterfactual_activations.py")
        _c6 = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_c6)
        bm, MAXPX = _c6.build_messages, 1024 * 1024

        @torch.inference_mode()
        def qpred(model, processor, row, kind, letter_ids):
            msgs = bm(row, kind, MAXPX)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            if imgs is not None and len(imgs) == 0:
                imgs = None
            inp = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
            li = int(inp.attention_mask.sum(1).item() - 1)
            k = len(row["options"])
            return int(model(**inp).logits[0, li][letter_ids[:k]].argmax().item())

        done = set()
        if out.exists():
            done = {json.loads(l)["candidate_id"] for l in out.read_text().splitlines() if l.strip()}
        pending = [r for cid, r in rowmap.items() if cid not in done]
        print(f"qwen-infer: {len(done)} done, {len(pending)} pending of {len(rowmap)}")
        if not pending:
            print("nothing to do"); return
        model, processor = load_qwen_vl(device=args.device)
        letter_ids = get_letter_token_ids(processor, args.device, n_letters=6)
        with out.open("a") as f:
            for r in tqdm(pending, desc="qwen 3-cond"):
                rec = {"candidate_id": r["candidate_id"], "source": r.get("source"),
                       "label": r["label"], "correct_index": int(r["correct_index"]),
                       "pred_vt": qpred(model, processor, r, "VT", letter_ids),
                       "pred_v":  qpred(model, processor, r, "V", letter_ids),
                       "pred_t":  qpred(model, processor, r, "T", letter_ids)}
                f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"qwen-infer: wrote -> {out}")
        return

    if args.model == "qwen":
        st = dr / "activations" / "qwen" / args.cf_subdir
        ll = np.load(st / "letter_logits.npy")
        idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
        pv, pvonly, pt = ll[:, 0].argmax(1), ll[:, 1].argmax(1), ll[:, 2].argmax(1)
        n = 0
        with out.open("w") as f:
            for i, r in enumerate(idx):
                cid = r["candidate_id"]; m = rowmap.get(cid, {})
                f.write(json.dumps({"candidate_id": cid, "source": m.get("source"),
                    "label": r["label"], "correct_index": int(r["correct_index"]),
                    "pred_vt": int(pv[i]), "pred_v": int(pvonly[i]), "pred_t": int(pt[i])}) + "\n")
                n += 1
        print(f"qwen: exported {n} items from cf-store -> {out}")
        return

    # llavanext: run predict_mcq
    from moground.models_llavanext import load_llavanext as load_m, predict_mcq
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["candidate_id"])
    pending = [r for cid, r in rowmap.items() if cid not in done]
    print(f"{args.model}: {len(done)} done, {len(pending)} pending of {len(rowmap)}")
    if not pending:
        print("nothing to do"); return
    model, processor = load_m(device=args.device)
    with out.open("a") as f:
        for r in tqdm(pending, desc=f"{args.model} 3-cond"):
            cap, img = r["caption_for_filter"], r["image_path"]
            rec = {"candidate_id": r["candidate_id"], "source": r.get("source"),
                   "label": r["label"], "correct_index": int(r["correct_index"]),
                   "pred_vt": predict_mcq(model, processor, r["question"], r["options"], image_path=img, caption=cap),
                   "pred_v":  predict_mcq(model, processor, r["question"], r["options"], image_path=img, caption=None),
                   "pred_t":  predict_mcq(model, processor, r["question"], r["options"], image_path=None, caption=cap)}
            f.write(json.dumps(rec) + "\n"); f.flush()
    print(f"{args.model}: wrote -> {out}")


if __name__ == "__main__":
    main()
