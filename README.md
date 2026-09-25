# MoGround

**Modality distraction in vision--language models: a certified benchmark, a measurement, and a
training-time repair.**

A VLM answers a question correctly from the image alone. Add a caption that is *certified to
contain no answer*, and the model changes its mind and gets it wrong. That event is **modality
distraction**, and measuring it requires knowing that the item was answerable from one modality in
the first place, which existing probes rarely establish. This repository builds the instrument,
measures the failure across seven backbones and four visual domains, and repairs it.

## What is here

**1. MoGround, a certified benchmark.** Every item is generated from a real image--caption pair
and then answered three times by an oracle, from the image alone (`V`), the text alone (`T`), and
both (`VT`). Only items whose answers match a single-modality signature survive: a vision-grounded
item is one the oracle gets right from `V` and `VT` but wrong from `T`, and symmetrically for
text. The gate is not a formality, **74.5% of generated candidates fail it**, and what remains is
3,418 items across natural photos (DCI), statistical charts (VisText), fine art (SemArt), and
radiology (ROCO), with a fixed 60/20/20 split. Two harder pools accompany it: a hand-written
subset (**MoGround-Human**, 125 items) and a companion pool assembled from A-OKVQA and RACE-high
that has natural headroom on both sides.

**2. The measurement.** Distraction is a *conditional* rate: `v-distraction = P(V+T wrong |
V-only correct)`, scored only on items a model demonstrably solves from its grounded modality, so
it always measures a capability the model has and loses. The headline findings:

* The widely reported text-over-vision asymmetry is **not universal**. Its direction and size track
  each model's grounding gap (`r = +0.86` over seven backbones); five of seven show the *reverse*
  on a ceiling-free pool.
* Distraction scales inversely with grounding strength in the target modality, `r = -0.90` over 70
  backbone x domain x modality cells, one line for every backbone, domain, and modality. The
  relation forecasts a held-out backbone's per-domain profile to within 1--2pp.
* The failure is consistent with **margin crossing**: an item flips when the irrelevant context
  pushes its single-modality answer margin below zero, and the answer crystallizes at a late,
  item-specific commit layer.

**3. The repair.** Because the certificate labels *which* modality carries each answer, it supplies
the supervision a fix needs. A LoRA adapter trained on MoGround's train split alone yields a
**robustness task vector** added at a single strength `w = 0.5`, chosen once on a selection side
that no reported number touches. It reduces v-distraction on **all seven backbones on all three
evaluation surfaces** (up to -73%), including pools the fine-tune never saw, at a mean general
capability cost of **0.1pp**. Inference-time alternatives, prompting, chain-of-thought,
SAE-feature ablation, dense direction ablation, and probe-gated activation patching, do not beat
their matched-random nulls.

## Setup

```bash
uv venv && uv sync && source .venv/bin/activate
```

API credentials (oracle generation and certification) live in `.credentials/`; GPU steps expect
`CUDA_VISIBLE_DEVICES` to be set and are best run under `tmux`.

## Using the released data

The curated release is `data/release/` (see its own `README.md` and `MANIFEST.json`): `moground/`
with per-item `pass_v` / `pass_t` / `pass_vt` certificates and the fixed splits,
`moground_human/`, `assembled/` post-audit, plus the 250 human-audit labels. To score your own
model, run each item three times (image only, text only, both) and condition as above; nothing in
the metric is specific to our backbones.

## Reproducing the pipeline

Scripts are numbered by pipeline stage; each has a docstring with its exact invocation.

| stage | scripts | what it does |
|---|---|---|
| build | `01_build_seeds*` -> `02_prepare_oracle_generation` -> `03_prepare_oracle_answer` | seed images, oracle question generation, three-condition answering |
| certify | `04_*` behavioral filters, `05_*` | keep only single-modality signatures |
| evaluate | `49_t8_collect`, `50_t8_score`, `61_behavioral_eval` | three-condition scoring for any backbone (`61` is the general entry point) |
| analyze | `54_grounding_strength`, `68_forecast_holdout`, `86_stats_hardening` | the grounding relation, leave-one-model-out forecasts, bootstrap CIs |
| mitigate | `76_adversarial_captions` -> `75_finetune_distraction` -> `78_benchmark_capability` | adversarial captions, LoRA training, capability evaluation |

Adding a backbone usually means one entry in the `REGISTRY` of `61_behavioral_eval.py`; the
analysis scripts glob over evaluation keys and pick it up.

## Caveats

* Certification is **operational**: single-modality answerability holds relative to the oracles
  used, not as absolute ground truth. A cross-oracle replication agrees at 94%, and a 250-item
  human audit passes 93.6%.
* Distraction rates on the Qwen2-VL family are sensitive to the Transformers version (dynamic
  resolution image handling); compare numbers only within a fixed environment. Ours is PyTorch
  2.11 / Transformers 5.6.
