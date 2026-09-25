#!/usr/bin/env python3
"""Universal h_V patching: apply the image-only residual at layer L to EVERY item, not just to
distracted ones, and score the whole pool.

tab:ablation's h_V patch is an oracle-gated positive control: it is applied only to items already
known to be distracted, which no deployable system knows in advance. The question here is whether
the same edit survives being applied unconditionally -- i.e. whether it is a usable intervention or
only a demonstration that the site is sufficient.

Prediction from the mechanism: patching h_V discards the caption's contribution, so it should help
vision-grounded items and destroy text-grounded ones. This script measures both sides.

    python scripts/98_universal_patch.py --model qwen --layer 28 --splits val,test
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_d87", Path(__file__).parent / "87_dense_control.py")
_d87 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d87)          # reuse predict_dense + loaders verbatim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--cf-subdir", default="dm_multidomain_v1_counterfactual_full")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--rows-file", default=None, help="explicit item jsonl (assembled pool)")
    ap.add_argument("--keep-ids", default=None, help="json with an id list to restrict to")
    ap.add_argument("--keep-key", default="heldout")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / a.config).read_text())
    dr = Path(cfg["paths"]["data_root"]); dm = Path(cfg["paths"]["dm_dir"])
    st = dr / "activations" / a.model / a.cf_subdir
    ll = np.load(st / "letter_logits.npy")
    acts = np.load(st / "activations.npy", mmap_mode="r")
    idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]

    rowmap = {}
    if a.rows_file:
        for l in Path(a.rows_file).read_text().splitlines():
            if l.strip():
                r = json.loads(l); rowmap[r["candidate_id"]] = r
    else:
        for sp in a.splits.split(","):
            p = dm / "splits" / f"{sp}.jsonl"
            if p.exists():
                for l in p.read_text().splitlines():
                    if l.strip():
                        r = json.loads(l); rowmap[r["candidate_id"]] = r
    if a.keep_ids:                            # e.g. held-out half, certified subset
        keep = set(json.loads(Path(a.keep_ids).read_text())[a.keep_key])
        viol = json.loads((dr / "diagnostics/assembled_caption_certification.json").read_text())["violations"]
        rowmap = {k: v for k, v in rowmap.items() if k in keep and k not in set(viol)}

    ci = np.array([r["correct_index"] for r in idx])
    lab = np.array([r["label"] for r in idx])
    v_ok = ll[:, 1].argmax(1) == ci          # image-only correct
    t_ok = ll[:, 2].argmax(1) == ci          # text-only correct
    vt_ok = ll[:, 0].argmax(1) == ci         # unpatched V+T correct

    sel = [i for i in range(len(idx)) if idx[i]["candidate_id"] in rowmap]
    if a.limit:
        sel = sel[:a.limit]
    print(f"{a.model} L{a.layer}: universal patch over {len(sel)} items "
          f"({(lab[sel]=='vision').sum()} vision / {(lab[sel]=='text').sum()} text)")

    _s56 = importlib.util.spec_from_file_location(
        "_s56", Path(__file__).parent / "56_causal_ablation.py")
    s56 = importlib.util.module_from_spec(_s56); _s56.loader.exec_module(s56)
    model, processor = s56.load_model(a.model, a.device); model.eval()
    letter_ids = _d87.get_letter_token_ids(processor, a.device)

    patched = np.zeros(len(idx), dtype=bool)
    for i in tqdm(sel, desc="universal-patch"):
        ev = torch.tensor(np.asarray(acts[i, 1, a.layer], dtype=np.float32), device=a.device)
        pred = _d87.predict_dense(model, processor, rowmap[idx[i]["candidate_id"]], a.layer,
                                  ev, "patch", letter_ids, a.device)
        patched[i] = (pred == ci[i])

    s = np.array(sel)
    isV = lab[s] == "vision"; isT = lab[s] == "text"
    def rates(ok):
        vs = isV & v_ok[s]; ts = isT & t_ok[s]
        return (float(ok[s].mean()),
                float((vs & ~ok[s]).sum() / max(vs.sum(), 1)),      # v-distraction
                float((ts & ~ok[s]).sum() / max(ts.sum(), 1)),      # t-distraction
                float(ok[s][isV].mean()), float(ok[s][isT].mean()))
    base = rates(vt_ok); pat = rates(patched)
    names = ["overall acc", "v-distraction", "t-distraction", "vision acc", "text acc"]
    print(f"\n=== universal h_V patch @ L{a.layer} ({a.model}, splits={a.splits}) ===")
    print(f"{'metric':16}{'unpatched':>12}{'patched':>12}{'delta':>12}")
    for n, b, p in zip(names, base, pat):
        print(f"{n:16}{b:>12.4f}{p:>12.4f}{p-b:>+12.4f}")
    out = dr / "diagnostics" / f"universal_patch_{a.model}_L{a.layer}_{('asm' + {'heldout': '', 'dev': 'dev'}.get(a.keep_key, a.keep_key)) if a.rows_file else a.splits.replace(',','')}.json"
    out.write_text(json.dumps({"model": a.model, "layer": a.layer, "splits": a.splits,
                               "n": len(sel), "unpatched": dict(zip(names, base)),
                               "patched": dict(zip(names, pat))}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
