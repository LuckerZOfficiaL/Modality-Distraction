"""Produce the corrected Phase-0 record for qwen3vl30b WITHOUT a GPU re-run.

Run 1 (2026-08-24, runs/logs/qwen3vl30b_phase0.log) verified everything GPU-dependent: loader
class, templates, BOS, letter ids, end-to-end forward, collector preflight (peak |a| = 65.0 ->
float16), one real LoRA training step (loss finite, peak 72.84GB decimal = 67.9GiB). Only the
LoRA-attachment AUDIT and its recording changed afterwards (explicit hard fields) -- and that
part is purely structural, so it runs here on the meta device with no card:

  * rebuild the model skeleton from config on device="meta";
  * resolve targets exactly as scripts/75 does, attach LoRA exactly as Phase 0 does;
  * assert: sites == 4 x n_LM_layers, all language-model attention, router/experts/vision absent;
  * cross-validate every value parsed from the run-1 log against the transcribed run-1 notes
    (rounding-aware) so the merged record cannot drift from what actually ran;
  * merge into one verdict json with per-section provenance. Any hard failure here is a REAL
    Phase-0 FAIL (exit 1, no verdict PASS written) -- same gate, zero GPU.

    python scripts/phase_phase0_meta_record.py
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, Qwen3VLMoeForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs/logs/qwen3vl30b_phase0.log"
OUT = ROOT / "runs/qwen3vl30b_phase0.json"
MODEL_KEY, MODEL_ID = "qwen3-vl-30b", "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Verbatim transcription of run 1's notes (the json this file replaces; values also echoed,
# rounded, in LOG). Kept as the exact-precision source; the log parse below cross-validates it.
RUN1_NOTES = {
    "model_id": MODEL_ID, "family": "qwen",
    "model_class": "Qwen3VLMoeForConditionalGeneration",
    "special_token_kwargs": {}, "template_has_bos": False, "n_leading_bos": 0, "bos_id": None,
    "sample_preds": [2, 3, 2],
    "peak_abs_hidden": 65.0, "recommended_act_dtype": "float16",
    "n_lora_targets": 4, "n_lora_sites_attached": 192,
    "train_loss": 0.0, "train_grad_norm": 3.6440315662744638e-06,
    "peak_mem_gb": 72.844222464,
}

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(f"{name}: {detail}")


def main() -> int:
    # ---- 1. cross-validate the run-1 log against the transcribed notes ----
    log = LOG.read_text()
    check("run-1 log records PHASE0_PASS (0 failures)", "PHASE0_PASS (0 failures)" in log)
    for pat, key, tol in [
        (r"peak=([0-9.]+) vs fp16", "peak_abs_hidden", 0.05),
        (r"peak CUDA memory during step: ([0-9.]+) GB", "peak_mem_gb", 0.05),
        (r"loss=([0-9.]+)", "train_loss", 5e-5),
        (r"grad-norm sum=([0-9.]+)", "train_grad_norm", 5e-5),
    ]:
        m = re.search(pat, log)
        ok = m and abs(float(m.group(1)) - RUN1_NOTES[key]) <= tol
        check(f"log/{key} agrees with transcribed run-1 value", bool(ok),
              f"log={m.group(1) if m else 'MISSING'} vs {RUN1_NOTES[key]}")
    m = re.search(r"-- (\d+) sites; bad=\[\]; non_attn=\[\]", log)
    check("run-1 attach audit: 192 sites, bad/non_attn empty",
          bool(m) and int(m.group(1)) == 192, m.group(0) if m else "line missing")
    check("run-1 BOS accounting: template False, leading 0",
          "template emits literal BOS: False; leading BOS in ids: 0" in log)
    check("run-1 loader class asserted",
          "got Qwen3VLMoeForConditionalGeneration, expected Qwen3VLMoeForConditionalGeneration" in log)

    # ---- 2. meta-device structural audit (the part that changed; no GPU) ----
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("m75", ROOT / "scripts/75_finetune_distraction.py")
    m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        model = Qwen3VLMoeForConditionalGeneration._from_config(cfg)
    targets = m75.resolve_targets(model, "qwen", "attn")
    check("targets resolve to the qwen bare-name set",
          set(targets) == {"q_proj", "k_proj", "v_proj", "o_proj"}, str(targets))

    attach_method = "get_peft_model(meta)"
    try:
        from peft import LoraConfig, get_peft_model
        tm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                              task_type="CAUSAL_LM", target_modules=targets))
        sites = [n for n, mm in tm.named_modules()
                 if hasattr(mm, "lora_A") and len(getattr(mm, "lora_A"))]
    except Exception as e:  # meta-attach not supported -> PEFT-equivalent suffix enumeration
        attach_method = f"suffix-enumeration fallback ({type(e).__name__})"
        sites = [n for n, _ in model.named_modules()
                 if any(n == t or n.endswith("." + t) for t in targets)]
    bad = [n for n in sites if any(t in n for t in ("visual", "vision_tower", "experts"))
           or n.endswith(".gate") or ".gate." in n]
    non_attn = [n for n in sites if "self_attn" not in n and "attn" not in n]
    n_layers = cfg.text_config.num_hidden_layers
    expected = 4 * n_layers
    check("meta audit: LoRA sites == 4 x n_LM_layers", len(sites) == expected,
          f"{len(sites)} vs {expected} (n_layers={n_layers}) via {attach_method}")
    check("meta audit: all sites are language-model attention", not non_attn, str(non_attn[:3]))
    check("meta audit: router/experts/vision absent from sites", not bad, str(bad[:3]))
    check("meta audit: site count equals run-1's live attach count",
          len(sites) == RUN1_NOTES["n_lora_sites_attached"], f"{len(sites)} vs 192")

    verdict = "PASS" if not FAIL else "FAIL"
    notes = dict(RUN1_NOTES)
    notes.update(n_lm_layers=n_layers, expected_lora_sites=expected,
                 sites_match_expected=len(sites) == expected,
                 router_or_expert_or_vision_in_sites=bool(bad),
                 non_attention_sites=bool(non_attn))
    out = {
        "model": MODEL_KEY, "model_id": MODEL_ID, "family": "qwen",
        "verdict": verdict, "failures": FAIL, "notes": notes,
        "provenance": {
            "gpu_checks": ("run 1, 2026-08-24 (log: runs/logs/qwen3vl30b_phase0.log, "
                           "mtime 08:14 local): loader/templates/BOS/letter-ids/forward/"
                           "collector-preflight/training-step incl. peak_mem_gb and the live "
                           "attach audit (192 sites, bad=[], non_attn=[])"),
            "structural_audit": (f"meta-device re-run {datetime.datetime.now(datetime.timezone.utc).isoformat()}, "
                                 f"no GPU, method: {attach_method}; run-1 log cross-validated "
                                 f"against transcribed exact values (rounding-aware)"),
            "why_no_gpu_rerun": ("GPU-dependent results unchanged from run 1; only the audit "
                                 "recording changed (coordinator decision 2026-08-24)"),
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n=== PHASE0_{verdict} (merged record) -> {OUT} ===")
    for f in FAIL:
        print(f"    - {f}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
