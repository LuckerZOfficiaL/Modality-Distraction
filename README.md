# Modality Distraction (MoGround)

Code for **MoGround: Measuring and Mitigating Modality Distraction in Vision-Language Models**.

A VLM answers a question correctly from the image alone. Add a caption that contains no answer, and
the model changes its mind and gets it wrong. That event is **modality distraction**. This
repository holds the pipeline that builds the benchmark, the evaluation harness that measures the
failure, and the mitigation that removes half of it.

## Released artifacts

| artifact | where |
|---|---|
| **Dataset** — all three pools, splits, human audits, image manifest | https://huggingface.co/datasets/LuckerZ/MoGround |
| **Sparse autoencoders** — 7 checkpoints for Qwen2.5-VL-3B and LLaVA-NeXT-8B | https://huggingface.co/LuckerZ/moground-saes |
| **Code** — this repository | https://github.com/LuckerZOfficiaL/Modality-Distraction |

The dataset ships no image bytes, because two of the six image sources forbid redistribution.
Every image is listed in `images/MANIFEST.jsonl` with its upstream identifier and sha256, and
`prepare_images.py` (in the dataset repo) rebuilds the image folders from the original sources and
verifies every checksum.

## The three pools

| pool | items | what it is |
|---|---|---|
| MoGround-Base | 3,418 | oracle-certified single-modality items over four visual domains (photos, charts, paintings, radiology), fixed 60/20/20 split |
| MoGround-Human | 125 | hand-authored hard subset, captions written to tempt a specific wrong option |
| MoGround-Retrieved | 4,757 | companion pool (A-OKVQA vision / RACE-high text) with cosine-retrieved cross-modal distractors, post-audit |

Every item is certified answerable from exactly one modality by a three-condition oracle check
(image alone, text alone, both), so the distraction it measures cannot be explained by the question
being unanswerable.

## How distraction is scored

Run each item three times: image only (V), context only (T), and both (V+T). Condition on the items
a model answers correctly from its grounding modality alone:

```
v-distraction = P(V+T wrong | V-only correct)   over vision-grounded items
t-distraction = P(V+T wrong | T-only correct)   over text-grounded items
```

The conditioning is model-relative on purpose: each model is scored on the items it itself solves,
so distraction always measures a capability the model demonstrably has and then loses.

## Setup

```bash
git clone https://github.com/LuckerZOfficiaL/Modality-Distraction
cd Modality-Distraction
uv sync                        # or: pip install -e .

# data
hf download LuckerZ/MoGround --repo-type dataset --local-dir data/moground
python data/moground/prepare_images.py --sources aokvqa cc3m     # see its docstring for the rest

# API oracles (only needed to rebuild the dataset, not to evaluate)
mkdir -p .credentials          # put your Gemini / Anthropic keys here; see scripts/oracle_smoke.py
```

## Layout

| path | what it does |
|---|---|
| `scripts/01-04*, build_*, oracle_*` | dataset construction: seeding, oracle generation, three-condition verification, splits |
| `scripts/61, 49, 18-19*, 78-79` | evaluation harness: the three-condition measurement, capability benchmarks |
| `scripts/54, 63-64, 45, 86, 94, 100-102, 124-125` | the grounding-strength relation, the margin mechanism, placebo and covariate controls |
| `scripts/75, 95-96, 99, 128, 135, 85, 87, 98, 103-106` | mitigation: the robustness vector and every baseline (prompting, chain-of-thought, M3ID, representation edits, patching) |
| `scripts/05, 08, build_pretraining_diversified, 06-14*` | SAE pretraining corpus, training, feature analysis |
| `scripts/108-109, 126-129` | human-audit tooling and analysis |
| `scripts/136-137` | builders that produced the released dataset and SAE bundles |
| `src/sae_steering/` | shared library: model loading, steering hooks, SAE, oracle clients |
| `configs/` | per-batch generation configs |

## Reproducing the headline numbers

1. Evaluate a backbone on a pool: `python scripts/61_behavioral_eval.py --model qwen --pool dm`
   (three conditions per item, writes one JSONL per run).
2. Fine-tune the robustness vector: `python scripts/75_finetune_distraction.py --model qwen`
   (LoRA rank 16 on attention only; the merged low-rank update is the vector).
3. Apply at strength `w` and re-evaluate: the vector enters the forward pass linearly, so
   `base + w·Δ` needs no retraining.
4. Baselines: `95_cot_baseline.py`, `96_prompt_baseline.py`, `128_m3id_baseline.py`, and the
   representation edits in `13_sae_steering.py` / `87_dense_control.py` / `98_universal_patch.py`.

## License

Code is released under the MIT license. The dataset annotations are CC BY-NC 4.0, and the image
sources keep their own terms — see the dataset card for the per-source table.
