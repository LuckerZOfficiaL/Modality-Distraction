"""Step 15b: assemble the per-feature interpretation notebook (interactive).

Reads:
  data/interpret/<model>/<variant>/layer<L>/steering_features/top3.json
  data/interpret/<model>/<variant>/layer<L>/steering_features/interpretations.json

Writes:
  data/interpret/<model>/<variant>/layer<L>/steering_features/notebook.ipynb

The notebook contains a small ipywidgets UI: pick modality (V or T) + rank
(1..N), and the kernel renders the chosen feature's stats + N max-activating
samples (images loaded from disk on demand and embedded inline as base64,
which sidesteps any broken notebook image renderer extension).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf
import yaml


def md(s: str) -> dict:
    return nbf.v4.new_markdown_cell(s)


def code(s: str) -> dict:
    return nbf.v4.new_code_cell(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--variant", default="diversified_v1")
    ap.add_argument("--layer", type=int, default=20)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(cfg["paths"]["data_root"])
    root = data_root / "interpret" / args.model / args.variant / f"layer{args.layer}" / "steering_features"

    data = json.load(open(root / "top3.json"))
    n_v = len(data["groups"][0]["features"])
    n_t = len(data["groups"][1]["features"])

    nb = nbf.v4.new_notebook()
    cells: list = []

    cells.append(md(
        "# SAE Steering Features — Input-Space Interpretation (interactive)\n\n"
        f"**Model**: `{data.get('model', args.model)}`. "
        f"**SAE**: layer {data['layer']}, variant `{data['sae_variant']}`.\n\n"
        f"**D_M (train_pass only)**: {data['n_train_pass']} rows "
        f"({data['n_vision']} vision-grounded + {data['n_text']} text-grounded).\n\n"
        "**Activation input is always multimodal**: every row is fed to Qwen as "
        "*caption + image + question + lettered options + 'Answer:'*. The V/T label "
        "describes only which modality the *correct answer* is grounded in (by MCQ design) — "
        "V = the question needs the image and the caption is uninformative; "
        "T = the question needs the caption and the image is uninformative. "
        "So the features below detect an *internal routing* signal (which side of the input "
        "the model leans on), not a difference in what it was shown.\n\n"
        f"**Feature selection**: {data['selection']}. These 40 latents (20 V + 20 T, "
        "ranked within each modality by |ρ|) are exactly the steering targets in step 13 "
        "(`scripts/13_sae_steering*.py`).\n\n"
        "**Stats glossary**: ρ = μᵥ − μₜ (signed modality preference, used for ranking); "
        "μᵥ / μₜ mean activations; fireᵥ / fireₜ firing rates; AUROC = rank separability of "
        "z against the V/T label (text=1, vision=0). AUROC > 0.5 → T-preferring, < 0.5 → V-preferring.\n\n"
        "Use the widget below to browse features one at a time. Per-feature interpretations "
        "are written by Sonnet 4.6 subagents that saw the 5 max-activating samples (image + "
        "question + options + caption) for each feature; Sonnet was explicitly instructed to "
        "flag weak / mixed features rather than confabulate a unifying label."
    ))

    setup = f'''import json, base64
from pathlib import Path
from IPython.display import HTML, display, clear_output
import ipywidgets as W

ROOT = Path({json.dumps(str(root))})
DATA = json.load(open(ROOT / "top3.json"))
INTERPS = {{(it["k"], it["group"]): it["interpretation"]
            for it in json.load(open(ROOT / "interpretations.json"))}}
GROUPS = {{g["group"]: g["features"] for g in DATA["groups"]}}
N_V = len(GROUPS["vision_preferring"])
N_T = len(GROUPS["text_preferring"])

def _img_html(path, width=420):
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    mime = {{"jpg":"jpeg","jpeg":"jpeg","png":"png","gif":"gif","webp":"webp"}}.get(ext, "jpeg")
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f\'<img src="data:image/{{mime}};base64,{{b64}}" width="{{width}}">\'

def _render(modality, rank):
    group = "vision_preferring" if modality == "V" else "text_preferring"
    feats = GROUPS[group]
    rank = max(1, min(rank, len(feats)))
    f = feats[rank - 1]
    k = f["k"]
    interp = INTERPS.get((k, group), "<em>(no interpretation)</em>")

    parts = []
    arrow = "↑V" if modality == "V" else "↑T"
    parts.append(
        f"<h2>#{{rank}} of {{len(feats)}} ({{arrow}}) — Feature k={{k}}</h2>"
        f"<p><b>Interpretation (Sonnet):</b> {{interp}}</p>"
        f"<table><tr><th>ρ</th><th>μᵥ</th><th>μₜ</th><th>fireᵥ</th><th>fireₜ</th><th>AUROC</th></tr>"
        f"<tr><td>{{f['rho']:+.3f}}</td><td>{{f['mu_v']:.3f}}</td><td>{{f['mu_t']:.3f}}</td>"
        f"<td>{{f['fire_v']:.3f}}</td><td>{{f['fire_t']:.3f}}</td><td>{{f['auroc']:.3f}}</td></tr></table>"
    )

    for i, s in enumerate(f["top_samples"], 1):
        opts = "".join(
            f"<li>{{'<b>' if j == s['correct_index'] else ''}}({{chr(65+j)}}) {{o}}"
            f"{{'</b>' if j == s['correct_index'] else ''}}</li>"
            for j, o in enumerate(s["options"] or [])
        )
        cap = (s.get("caption") or "").strip()
        cap_block = ""
        if cap:
            cap_role = ("answer is grounded in this caption" if s["label"] == "text"
                        else "caption is uninformative for the answer; question requires the image")
            cap_block = f"<p><b>Caption</b> (<i>{{cap_role}}</i>): {{cap}}</p>"
        parts.append(
            f"<hr><h4>Sample {{i}} — <code>{{s['candidate_id']}}</code> "
            f"(label=<code>{{s['label']}}</code>, source=<code>{{s.get('source','?')}}</code>, "
            f"act = <b>{{s['activation']:.3f}}</b>)</h4>"
            f"{{_img_html(s['image_path'])}}"
            f"<p><b>Q:</b> {{s['question']}}</p><ul>{{opts}}</ul>{{cap_block}}"
        )
    display(HTML("".join(parts)))

modality_w = W.ToggleButtons(options=[("Vision-preferring (↑V)", "V"),
                                      ("Text-preferring (↑T)", "T")], value="V",
                             description="Modality:")
rank_w = W.BoundedIntText(value=1, min=1, max=max(N_V, N_T), step=1, description="Rank #:")
out_w = W.Output()

def _on_change(_=None):
    cap = N_V if modality_w.value == "V" else N_T
    if rank_w.max != cap:
        rank_w.max = cap
    if rank_w.value > cap:
        rank_w.value = cap
    with out_w:
        clear_output(wait=True)
        _render(modality_w.value, rank_w.value)

modality_w.observe(_on_change, names="value")
rank_w.observe(_on_change, names="value")
display(W.HBox([modality_w, rank_w]), out_w)
_on_change()
'''
    cells.append(code(setup))

    nb["cells"] = cells
    out_path = root / "notebook.ipynb"
    nbf.write(nb, out_path)
    print(f"wrote {out_path}  ({len(cells)} cells, {n_v} V + {n_t} T features browseable)")


if __name__ == "__main__":
    main()
