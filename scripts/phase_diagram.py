"""Correctability phase diagram: for each backbone, trace decoupling (law-residual) and general-
capability cost as the LoRA intervention-strength coefficient w sweeps. Order the backbones by base
grounding and ask whether the CLEAN window (decouple at cost<eps) shrinks/vanishes with grounding."""
import json, collections
import numpy as np
from pathlib import Path

RT = Path(__file__).resolve().parent.parent
tids = {json.loads(l)["candidate_id"] for l in (RT / "data/dm/multidomain_v1/splits/test.jsonl").read_text().splitlines() if l.strip()}
pts = json.loads((RT / "data/diagnostics/grounding_strength.json").read_text())
gv = np.array([p["ground"] for p in pts if p["modality"] == "vision"]); dv = np.array([p["distr"] for p in pts if p["modality"] == "vision"])
B, A = np.polyfit(gv, dv, 1)
BEN = ["mmstar", "mmbench", "seedbench", "scienceqa", "naturalbench"]
WS = ["0.0", "0.5", "0.75", "1.0", "1.5", "2.0"]
BB = [("mistral", ), ("qwen2b", "Qwen2-VL-2B"), ("next", "LLaVA-NeXT-8B"),
      ("qwen3b", "Qwen2.5-VL-3B"), ("qwen7b", "Qwen2.5-VL-7B")]  # will re-sort by measured base grounding
EPS = 0.02  # capability-cost tolerance; decoupled = residual < -0.02


def decouple(key):
    rs = [json.loads(l) for l in (RT / f"data/eval/t8/{key}.jsonl").read_text().splitlines() if l.strip()]
    by = collections.defaultdict(list)
    for r in rs:
        if r["candidate_id"] in tids and r["label"] == "vision":
            by[r["source"]].append(r)
    gg, dd = [], []
    for dom, it in by.items():
        if len(it) < 20:
            continue
        ci = np.array([r["correct_index"] for r in it]); v = np.array([r["pred_v"] for r in it]); vt = np.array([r["pred_vt"] for r in it])
        vs = v == ci; gg.append(vs.mean()); dd.append(((vs) & (vt != ci)).sum() / max(vs.sum(), 1))
    rs = [json.loads(l) for l in (RT / f"data/eval/t8_assembled/{key}.jsonl").read_text().splitlines() if l.strip()]
    ci = np.array([r["correct_index"] for r in rs]); lab = np.array([r["label"] for r in rs])
    v = np.array([r["pred_v"] for r in rs]); vt = np.array([r["pred_vt"] for r in rs]); t = np.array([r["pred_t"] for r in rs])
    m = lab == "vision"; vs = m & (v == ci); gg.append((v[m] == ci[m]).mean()); dd.append((vs & (vt != ci)).sum() / max(vs.sum(), 1))
    g = np.array(gg); d = np.array(dd); T = lab == "text"
    return dict(vdistr=d.mean(), resid=(d - (A + B * g)).mean(), capuse=(vt[T] == ci[T]).mean())


def cap(tag, w):
    return {b: json.load(open(RT / f"data/diagnostics/capability/{b}_phase_{tag}_w{w}.json"))["acc"] for b in BEN}


rows = []
for tag, name in BB:
    base_cap = cap(tag, "0.0")
    base_g = decouple(f"{tag}_w0.0")["vdistr"]  # placeholder; use base grounding below
    traj = []
    for w in WS:
        dec = decouple(f"{tag}_w{w}")
        c = cap(tag, w)
        C = np.mean([c[b] - base_cap[b] for b in BEN])
        traj.append(dict(w=float(w), resid=dec["resid"], vdistr=dec["vdistr"], capuse=dec["capuse"], C=C))
    # base grounding proxy = base caption-use (text-side grounding, the axis correctability tracked)
    rows.append((tag, name, traj[0]["capuse"], traj))

rows.sort(key=lambda r: r[2])  # order by base caption-use (grounding)
print(f"8-model law resid; decoupled = resid<-{EPS}; CLEAN cell = decoupled AND C>-{EPS}\n")
print("Per-backbone trajectory (ordered by base grounding = base caption-use):")
for tag, name, bg, traj in rows:
    print(f"\n{name}  (base grounding {bg:.3f})")
    print(f"  {'w':>5}{'resid':>9}{'vdistr':>8}{'capUse':>8}{'C(5bm)':>9}  phase")
    window = []
    for t in traj:
        decoupled = t["resid"] < -EPS
        clean = decoupled and t["C"] > -EPS
        ph = "CLEAN" if clean else ("DESTRUCT" if decoupled else "inert")
        if clean:
            window.append(t["w"])
        print(f"  {t['w']:>5.2f}{t['resid']:>+9.3f}{t['vdistr']:>8.3f}{t['capuse']:>8.3f}{t['C']*100:>+8.1f}  {ph}")
    # D* = max decoupling (most negative resid) achievable at cost < EPS
    ok = [t for t in traj if t["C"] > -EPS]
    Dstar = min([t["resid"] for t in ok]) if ok else 0.0
    print(f"  -> clean window w in {window if window else '(none)'};  D* (best decouple @cost<2pp) = {Dstar:+.3f}")

print("\n=== HEADLINE SCATTER: does clean-correction track grounding? ===")
print(f"{'backbone':18}{'base_grnd':>10}{'D*':>9}{'clean_ws':>10}")
xs, ys = [], []
for tag, name, bg, traj in rows:
    ok = [t for t in traj if t["C"] > -EPS]
    Dstar = min([t["resid"] for t in ok]) if ok else 0.0
    ws = [t["w"] for t in traj if t["resid"] < -EPS and t["C"] > -EPS]
    xs.append(bg); ys.append(-Dstar)  # -Dstar: larger = more clean decoupling
    print(f"{name:18}{bg:>10.3f}{Dstar:>+9.3f}{(len(ws)):>10}")
if len(set(xs)) > 1:
    r = np.corrcoef(xs, ys)[0, 1]
    print(f"\ncorr(base grounding, clean-decoupling D*) = {r:+.3f}  (n={len(xs)} backbones)")
