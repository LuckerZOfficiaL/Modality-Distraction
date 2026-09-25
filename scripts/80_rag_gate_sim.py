"""Phase-0 OFFLINE simulation: margin-gated context use on a mixed multimodal-RAG stream.

Regime: a stream where retrieved text context is ESSENTIAL for some items (T-grounded: answer only in
the context) and IRRELEVANT for others (V-grounded: answer in the image; context can only distract).
Neither trivial policy is good here: always-use suffers v-distraction on V-items; never-use collapses
T-items to their V-only floor. The margin gate uses the model's own signals (no labels): if the
no-context answer is confident (V-only margin >= tau) and the context CHANGES the answer, distrust the
context; otherwise use it.

Retrieval-quality dial q: each T-item's needed context is retrieved with prob q, else missing
(approximated by the V-only cell; the misleading-retrieval cell needs new forwards -> Phase 1).
V-items always carry their (irrelevant) context.

Pre-registered kill: the gate must beat BOTH always-use and never-use on the held-out test split at
realistic q. tau is selected on train+val only. Logit-capable models only (margin needs logits).

    python scripts/80_rag_gate_sim.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RT = Path(__file__).resolve().parent.parent
NAME = {"qwen": "Qwen2.5-VL-3B", "Qwen_Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "Qwen_Qwen2-VL-2B-Instruct": "Qwen2-VL-2B", "OpenGVLab_InternVL3-8B-hf": "InternVL3-8B",
        "llava-hf_llava-1.5-7b-hf": "LLaVA-1.5-7B", "llava-hf_llava-onevision-qwen2-7b-ov-hf": "LLaVA-OV-7B"}
QS = [1.0, 0.75, 0.5, 0.25, 0.0]
TAUS = np.arange(0.0, 8.1, 0.25)
SEEDS = range(5)          # retrieval-miss resampling


def splits():
    out = {}
    for sp in ("train", "val", "test"):
        for l in (RT / f"data/dm/multidomain_v1/splits/{sp}.jsonl").read_text().splitlines():
            if l.strip():
                out[json.loads(l)["candidate_id"]] = sp
    return out


def load_model(key):
    """rows with correct_index, label, pred_v/vt, margin of V-only logits."""
    rows = []
    if key == "qwen":  # logits live in the cf-store
        st = RT / "data/activations/qwen/dm_multidomain_v1_counterfactual_full"
        ll = np.load(st / "letter_logits.npy")  # (N,3,4) VT,V,T
        idx = [json.loads(l) for l in (st / "index.jsonl").read_text().splitlines() if l.strip()]
        for i, r in enumerate(idx):
            lv = np.sort(ll[i, 1])[::-1]
            rows.append(dict(cid=r["candidate_id"], label=r["label"], ci=r["correct_index"],
                             pv=int(ll[i, 1].argmax()), pvt=int(ll[i, 0].argmax()),
                             mv=float(lv[0] - lv[1])))
        return rows
    for l in (RT / f"data/eval/t8/{key}.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if "logits_v" not in r:
            return []
        lv = np.sort(np.array(r["logits_v"], dtype=float))[::-1]
        rows.append(dict(cid=r["candidate_id"], label=r["label"], ci=r["correct_index"],
                         pv=r["pred_v"], pvt=r["pred_vt"], mv=float(lv[0] - lv[1])))
    return rows


def stream_acc(rows, q, tau, rng, policy):
    """Accuracy of a policy on the stream at retrieval quality q.

    Per item: T-item context retrieved w.p. q (else missing -> V-only cell for every policy).
    Policies: always (use context when present), never (ignore context), oracle (use iff T-item w/ gold),
    gate (use unless no-context answer is confident AND context flips it).
    """
    ok = 0
    for r in rows:
        has_ctx = True if r["label"] == "vision" else (rng.random() < q)
        if not has_ctx:
            pred = r["pv"]
        elif policy == "always":
            pred = r["pvt"]
        elif policy == "never":
            pred = r["pv"]
        elif policy == "oracle":
            pred = r["pvt"] if r["label"] == "text" else r["pv"]
        else:  # gate
            distrust = (r["pv"] != r["pvt"]) and (r["mv"] >= tau)
            pred = r["pv"] if distrust else r["pvt"]
        ok += int(pred == r["ci"])
    return ok / len(rows)


def main():
    sp = splits()
    print(f"stream = full D_M mix (V irrelevant-context / T essential-context); tau from train+val, report on TEST")
    print(f"kill: gate must beat BOTH always and never at each q\n")
    summary = {}
    for key, name in NAME.items():
        rows = load_model(key)
        if not rows:
            print(f"{name}: no logits, skipped"); continue
        dev = [r for r in rows if sp.get(r["cid"]) in ("train", "val")]
        test = [r for r in rows if sp.get(r["cid"]) == "test"]
        # select tau on dev at q=0.5 (mid retrieval quality), avg over seeds
        best_tau, best = None, -1
        for tau in TAUS:
            acc = np.mean([stream_acc(dev, 0.5, tau, np.random.default_rng(s), "gate") for s in SEEDS])
            if acc > best:
                best, best_tau = acc, tau
        print(f"### {name}  (test n={len(test)}, tau*={best_tau:.2f} from dev)")
        print(f"  {'q':>5} {'always':>8} {'never':>8} {'gate':>8} {'oracle':>8}  verdict")
        wins = 0
        for q in QS:
            accs = {}
            for pol in ("always", "never", "gate", "oracle"):
                accs[pol] = np.mean([stream_acc(test, q, best_tau, np.random.default_rng(100 + s), pol)
                                     for s in SEEDS])
            beat = accs["gate"] > max(accs["always"], accs["never"])
            wins += int(beat)
            print(f"  {q:>5.2f} {accs['always']:>8.3f} {accs['never']:>8.3f} {accs['gate']:>8.3f} "
                  f"{accs['oracle']:>8.3f}  {'GATE WINS' if beat else '-'}")
        summary[name] = wins
        print()
    print("=== gate beats both baselines at N of 5 retrieval-quality levels ===")
    for n, w in summary.items():
        print(f"  {n:16} {w}/5")


if __name__ == "__main__":
    main()
