"""Appendix gallery: for a few auto-interpreted SAE distraction features per backbone, show the
COMPLETE illustrative item (image + question + options + caption). Uses the Sonnet label + its chosen
best example (matched back to the enriched dump from scripts/47 for the full item + image).

    python scripts/74_appendix_sae_features.py
"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DR = ROOT / "data"
SAED = DR / "diagnostics/sae_distraction"
FIGS = ROOT / "paper/figs/sae_features"; FIGS.mkdir(parents=True, exist_ok=True)
LAB = json.loads((SAED / "autointerp_labels.json").read_text())
DOM = {"dci": "natural photo", "vistext": "statistical chart", "semart": "fine-art painting",
       "roco": "medical radiology"}
# (display, dump_path, feature ids) — clearest distraction-associated features per backbone
# Shown features are the interpretably-labelled ones with the highest distraction SELECTIVITY
# (firing rate on distracted minus on robust V-solvable items, from the *_top.json dumps).
# Features whose selectivity is ~0 fire on essentially every item and are domain, not distraction,
MODELS = [("Qwen2.5-VL-3B", SAED / "layer28_examples.json", [9735, 653]),
          ("LLaVA-NeXT-8B", SAED / "llavanext/layer18_examples.json", [30527, 4702])]


def esc(s):
    s = (s or "").replace("\n", " ").strip()
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def trunc(s, n):
    w = (s or "").split()
    return esc(" ".join(w[:n])) + (r"\,\dots" if len(w) > n else "")


def pick_example(exs, target_q):
    """Prefer an actually-DISTRACTED item: the gallery illustrates distraction events, so showing a
    top-activating item the model answers correctly would misrepresent the feature. Among distracted
    items we keep Sonnet's chosen example if it is one, else the highest-activating distracted item;
    only if the feature has no distracted example at all do we fall back to the top activation."""
    distracted = [e for e in exs if e.get("distracted")]
    pool = distracted or exs
    for e in pool:  # match Sonnet's chosen best example by question
        if (e.get("question") or "").strip()[:60] == (target_q or "").strip()[:60]:
            return e
    return pool[0]  # highest-act within the pool


def main():
    out = []
    for disp, path, feats in MODELS:
        dump = json.loads(path.read_text())
        top = {str(f["feature"]): f for f in
               json.loads(Path(str(path).replace("_examples.json", "_top.json")).read_text())["top_features"]}
        out.append(r"\subsection{" + disp + r"}")
        labels = LAB.get(disp, {})
        for fid in feats:
            fj = dump[str(fid)]; lv = labels.get(str(fid), {})
            e = pick_example(fj["examples"], lv.get("best_example", {}).get("question", ""))
            tag = f"{disp.replace('.', '').replace('-', '').replace(' ', '')}_{fid}"
            im = Image.open(e["image_path"]).convert("RGB"); im.thumbnail((440, 440))
            im.save(FIGS / f"{tag}.jpg", quality=85)
            LET = "ABCD"
            SEP = r"\\[3pt]\rule{\linewidth}{0.15pt}\\[3pt]"
            opts = " \\quad ".join(
                (r"\textbf{" + f"{LET[i]}. " + esc(o) + "}") if i == e.get("correct_index")
                else f"{LET[i]}. " + esc(o) for i, o in enumerate(e["options"]))
            out.append(
                # violet hue: a different kind of exhibit from the dataset examples (cyan)
                r"\begin{tcolorbox}[colback=violet!4,colframe=violet!35,arc=1mm,boxrule=0.3pt,"
                r"width=\textwidth,top=3pt,bottom=3pt,left=5pt,right=5pt]" "\n"
                r"\textbf{Feature \#" + str(fid) + r"}: \emph{" + esc(lv.get("label", "")) + r"}\\[2pt]"
                r"{\small fires on " + f"{top[str(fid)]['fr_distracted']:.2f}" + r" of distracted vs.\ "
                + f"{top[str(fid)]['fr_robust']:.2f}" + r" of robust V-solvable items "
                r"(selectivity $" + f"{top[str(fid)]['fr_distracted'] - top[str(fid)]['fr_robust']:+.2f}" +
                r"$); activation " + f"{e['act']:.1f}" + r" on the item below.}" "\n"
                r"\tcbline" "\n"
                r"\noindent\begin{minipage}[t]{0.30\linewidth}\centering\vspace{0pt}"
                r"\includegraphics[width=\linewidth,height=3.4cm,keepaspectratio]"
                r"{sae_features/" + tag + r".jpg}\end{minipage}\hfill" "\n"
                r"\begin{minipage}[t]{0.66\linewidth}\small\vspace{0pt}" "\n"
                r"\textbf{Domain.}~" + DOM.get(e["source"], e["source"] or "") + SEP + "\n"
                r"\textbf{Question.}~" + esc(e["question"]) + SEP + "\n"
                r"\textbf{Options.}~" + opts + SEP + "\n"
                r"\textbf{Caption.}~" + trunc(e["caption"], 40) + "\n"
                r"\end{minipage}" "\n"
                r"\end{tcolorbox}")
    (ROOT / "paper/tables/appendix_sae_features.tex").write_text("\n".join(out) + "\n")
    print(f"wrote appendix_sae_features.tex ({sum(len(f) for _,_,f in MODELS)} feature items, images in {FIGS})")


if __name__ == "__main__":
    main()
