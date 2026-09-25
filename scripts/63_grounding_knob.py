"""#2 Causal grounding-knob sweep: turn the grounding-strength law from
correlational-across-models into causal-within-a-model.

Take ONE model (Qwen2.5-VL-3B) and degrade its VISION grounding in controlled steps by
Gaussian-blurring the image (radius 0,1,2,4,8,16 -- token count fixed, purely removes
visual information). At each level, measure grounding strength (V-only accuracy) and
v-distraction on D_M vision items. If v-distraction climbs along the SAME
(grounding, distraction) curve traced by the cross-model law, grounding strength
*causally governs* distraction (within a fixed model), not just co-varies with it.

Confound note: blur also raises item difficulty; we plot in (grounding, distraction)
space and overlay the cross-model law -- the claim is that the trajectory follows that
curve, i.e. grounding is the mediating variable however it is lowered.

GPU. Emits trajectory json + paper/figs/grounding_knob.pdf (overlay on the law).

    CUDA_VISIBLE_DEVICES=0 python scripts/63_grounding_knob.py --limit 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from sae_steering.models import load_qwen_vl
from sae_steering.steering import get_letter_token_ids

MAXPX = 1024 * 1024


def degrade(img, knob, level):
    """Mechanistically-distinct vision-grounding knobs:
       blur      = Gaussian blur radius `level` (removes high-freq info; token count fixed)
       downscale = resize to fraction `level` (reduces the visual-token budget)"""
    if knob == "blur":
        return img if level <= 0 else img.filter(ImageFilter.GaussianBlur(radius=level))
    if level >= 1.0:
        return img
    w, h = img.size
    return img.resize((max(1, int(w * level)), max(1, int(h * level))))


def build(img, row, use_caption):
    content = [{"type": "image", "image": img, "max_pixels": MAXPX}]
    opts = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", row["options"]))
    txt = (f"Caption: {row['caption_for_filter']}\n" if use_caption else "")
    txt += f"Question: {row['question']}\n{opts}\nReply with one letter (A-D)."
    content.append({"type": "text", "text": txt})
    return [{"role": "user", "content": content}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_multidomain_v1.yaml")
    ap.add_argument("--knob", default="blur", choices=["blur", "downscale"])
    ap.add_argument("--levels", default=None, help="comma list; default per knob")
    ap.add_argument("--limit", type=int, default=400, help="vision items sampled (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm = Path(cfg["paths"]["dm_dir"])
    rows = []
    for sp in ("train", "val", "test"):
        p = dm / "splits" / f"{sp}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["label"] == "vision"]
    rng = np.random.default_rng(args.seed); rng.shuffle(rows); rows = rows[:args.limit]
    DEFAULT = {"blur": "0,1,2,4,8,16", "downscale": "1.0,0.7,0.5,0.35,0.25,0.15"}
    levels = [float(x) for x in (args.levels or DEFAULT[args.knob]).split(",")]
    print(f"vision items={len(rows)}  knob={args.knob}  levels={levels}")

    model, processor = load_qwen_vl(device=args.device)
    lids = get_letter_token_ids(processor, args.device, n_letters=6)

    @torch.inference_mode()
    def pred(img, row, use_caption):
        msgs = build(img, row, use_caption)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
        li = int(inp.attention_mask[0].sum().item() - 1); k = len(row["options"])
        return int(model(**inp).logits[0, li][lids[:k]].argmax().item())

    traj = []
    for lv in levels:
        pv, pvt, ci = [], [], []
        for r in tqdm(rows, desc=f"{args.knob}={lv}", leave=False):
            img = degrade(Image.open(r["image_path"]).convert("RGB"), args.knob, lv)
            pv.append(pred(img, r, False)); pvt.append(pred(img, r, True)); ci.append(r["correct_index"])
        pv, pvt, ci = map(np.array, (pv, pvt, ci))
        vok = pv == ci
        ground = float(vok.mean())
        vdist = float(((pv == ci) & (pvt != ci)).sum() / max(vok.sum(), 1))
        vt_acc = float((pvt == ci).mean())
        traj.append({"level": lv, "grounding": round(ground, 4), "v_distraction": round(vdist, 4),
                     "vt_acc": round(vt_acc, 4), "n": len(rows)})
        print(f"  {args.knob}={lv:>5}: grounding(V-only)={ground:.3f}  v-distraction={vdist:.3f}  V+T acc={vt_acc:.3f}")

    out = Path(cfg["paths"]["data_root"]) / "diagnostics" / f"grounding_knob_qwen_{args.knob}.json"
    out.write_text(json.dumps({"model": "qwen2.5-vl-3b", "knob": args.knob, "trajectory": traj}, indent=2))
    print(f"-> {out}")

    # overlay figure: ALL within-model knob trajectories on the cross-model law (convergent evidence)
    try:
        import glob, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ddir = Path(cfg["paths"]["data_root"]) / "diagnostics"
        law = json.loads((ddir / "grounding_strength.json").read_text())
        lg = [c["ground"] for c in law if c["modality"] == "vision"]
        ld = [c["distr"] for c in law if c["modality"] == "vision"]
        colors = {"blur": "tab:red", "gaussian_blur": "tab:red", "downscale": "tab:blue"}
        fig, ax = plt.subplots(figsize=(5.2, 3.7))
        ax.scatter(lg, ld, c="0.7", s=26, label="cross-model law (vision cells)")
        for jf in sorted(glob.glob(str(ddir / "grounding_knob_qwen*.json"))):
            dj = json.loads(Path(jf).read_text()); kn = dj.get("knob", "blur"); tr = dj["trajectory"]
            ax.plot([t["grounding"] for t in tr], [t["v_distraction"] for t in tr], "-o",
                    color=colors.get(kn, "tab:green"), label=f"Qwen2.5-VL-3B, {kn} sweep")
            for t in tr:
                lv = t.get("level", t.get("radius"))
                ax.annotate(f"{lv:g}", (t["grounding"], t["v_distraction"]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("grounding strength (V-only accuracy)"); ax.set_ylabel("v-distraction")
        ax.set_title("Within-model vision degradation traces the cross-model law")
        ax.legend(frameon=False, fontsize=7); fig.tight_layout()
        fp = Path(__file__).resolve().parent.parent / "paper/figs/grounding_knob.pdf"
        fig.savefig(fp); plt.close(fig); print(f"-> {fp}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
