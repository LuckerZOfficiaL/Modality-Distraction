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
| `scripts/build_seeds*, prepare_oracle_*, oracle_*, vlm_behavioral_filter*, build_*` | dataset construction: seeding, oracle generation, three-condition verification, splits |
| `scripts/behavioral_eval.py, t8_collect.py, canonical_eval.py, benchmark_*, prep_*` | evaluation harness: the three-condition measurement, capability benchmarks |
| `scripts/grounding_strength.py, grounding_knob*, margin_model.py, logit_lens_distraction.py, *_controls.py, placebo_*` | the grounding-strength relation, the margin mechanism, placebo and covariate controls |
| `scripts/finetune_distraction.py, cot_*, prompt_baseline.py, m3id_*, gated_*, dense_control.py, universal_patch.py, gate*` | mitigation: the robustness vector and every baseline |
| `scripts/train_sae.py, download_cc3m_subset.py, build_pretraining_diversified.py, collect_*_activations*, sae_*, identify_features*, train_probes*` | SAE pretraining corpus, training, feature analysis |
| `scripts/validation_analysis.py, *audit*` | human-audit tooling and analysis |
| `scripts/build_public_release.py, build_sae_release.py` | builders that produced the released dataset and SAE bundles |
| `src/moground/` | shared library: model loading, steering hooks, SAE, oracle clients |
| `configs/` | per-batch generation configs |

## Getting the numbers

1. Evaluate a backbone on a pool: `python scripts/behavioral_eval.py --model qwen --pool dm`
   (three conditions per item, writes one JSONL per run).
2. Fine-tune the robustness vector: `python scripts/finetune_distraction.py --model qwen`
   (LoRA rank 16 on attention only; the merged low-rank update is the vector).
3. Apply at strength `w` and re-evaluate: the vector enters the forward pass linearly, so
   `base + w·Δ` needs no retraining.
4. Baselines: `cot_baseline.py`, `prompt_baseline.py`, `m3id_baseline.py`, and the
   representation edits in `sae_steering.py` / `dense_control.py` / `universal_patch.py`.

Each run writes per-item predictions as JSONL. Computing the distraction rates from them is the
two-line conditional above. No table or figure rendering ships here.

## License

Code is released under the MIT license. The dataset annotations are CC BY-NC 4.0, and the image
sources keep their own terms — see the dataset card for the per-source table.
