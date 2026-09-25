"""Shared infrastructure for the three steering-comparison scripts:
probe direction (11), contrastive direction (12), SAE multiplicative (13).

Each script supplies its own steering vector / hook; everything else
(eval pool selection, baseline phase, hooked phase, 2x2 summary) lives here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


POOL_CHOICES = ("test", "unseen", "fails", "all")


def fix_negative_args(flags=("--alphas",)) -> None:
    """Glue `--alphas` to its value in sys.argv so argparse accepts e.g.
    `--alphas -3,-1,1,3` (which argparse otherwise treats as an unknown flag
    because the value's `-` prefix doesn't match argparse's negative-number
    regex when commas are present). Call before parser.parse_args()."""
    import sys
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok in flags and i + 1 < len(sys.argv) and sys.argv[i + 1].startswith("-"):
            new_argv.append(f"{tok}={sys.argv[i + 1]}")
            i += 2
        else:
            new_argv.append(tok)
            i += 1
    sys.argv = new_argv


CANONICAL_MAX_PIXELS = 1024 * 1024  # matches 06c counterfactual collection setup
CANONICAL_CF_SUBDIR = "dm_multidomain_v1_counterfactual"


LETTERS = ["A", "B", "C", "D", "E", "F"]


def build_messages(row: dict, max_pixels: int = CANONICAL_MAX_PIXELS) -> list[dict]:
    """Canonical Qwen prompt with image capped at ``max_pixels`` (default 1024*1024).

    All steering / baseline / patching scripts go through this function, so any
    forward pass produced via the project's helpers is aligned with the
    cf-store baseline produced by ``collect_dm_counterfactual_activations.py``.
    Do not change the default without invalidating prior canonical numbers.

    Supports a variable number of options (``row["options"]`` of length 2--6) and
    an optional text/caption modality: if ``row["caption_for_filter"]`` is absent
    or empty, the caption block is dropped (external MCQ benchmarks have no
    separable text modality). For 4-option rows carrying a caption -- i.e. every
    multidomain $D_M$ row -- the produced text is byte-identical to the prior
    hard-coded prompt, so canonical numbers are preserved.
    """
    n = len(row["options"])
    letters = LETTERS[:n]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    if n == 2:
        letter_list = f"{letters[0]} or {letters[1]}"
    else:
        letter_list = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    content = []
    caption = row.get("caption_for_filter")
    if caption:
        content.append({"type": "text", "text": f"Caption: {caption}"})
    content.append({"type": "image", "image": row["image_path"], "max_pixels": max_pixels})
    # optional steering-by-prompt directive (e.g. "Focus on the image."); absent
    # for every D_M row, so the canonical prompt is unchanged.
    directive = row.get("directive")
    dir_line = f"{directive}\n" if directive else ""
    content.append({"type": "text", "text": (
        f"Question: {row['question']}\n{opts}\n\n{dir_line}"
        f"Reply with exactly one letter ({letter_list}) and nothing else."
    )})
    return [{"role": "user", "content": content}]


def load_canonical_baseline(
    data_root, cf_subdir: str = CANONICAL_CF_SUBDIR,
) -> tuple[dict, dict, dict]:
    """Read the canonical V+T baseline from the cf-store letter_logits.

    Returns three cid-keyed dicts (one per condition VT / V / T), each mapping
    candidate_id -> argmax letter index. Use this as the single source of
    truth for "what does the model answer" -- the cf-store is produced by 06c
    with the canonical prompt setup, so its argmax matches any forward pass
    produced via this module's helpers (modulo the ~1.4% irreducible
    batching-padding numerical floor).
    """
    cf_dir = Path(data_root) / "activations" / "qwen" / cf_subdir
    logits = np.load(cf_dir / "letter_logits.npy")   # (N, 3, 4) [VT, V, T]
    rows = [json.loads(l) for l in (cf_dir / "index.jsonl").read_text().splitlines()
            if l.strip()]
    baseline_pred, v_only_pred, t_only_pred = {}, {}, {}
    for i, r in enumerate(rows):
        cid = r["candidate_id"]
        baseline_pred[cid] = int(logits[i, 0].argmax())
        v_only_pred[cid]   = int(logits[i, 1].argmax())
        t_only_pred[cid]   = int(logits[i, 2].argmax())
    return baseline_pred, v_only_pred, t_only_pred


class SteeringHook:
    """Generic forward hook. Subclasses override edit(h).

    If `broadcast` is False (default), edit() receives the last-token slice
    [B, D] and the result is written back at the last position.

    If `broadcast` is True, edit() receives all tokens flattened to [B*S, D]
    and the result replaces the entire sequence. Subclasses don't need to
    distinguish — they operate on a [..., D] tensor either way."""

    def __init__(self, broadcast: bool = False):
        self.last_idx: int | None = None
        self.broadcast = broadcast

    def edit(self, h: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if self.last_idx is None:
            return output
        if self.broadcast:
            B, S, D = h.shape
            new = self.edit(h.reshape(B * S, D))
            if new is None:
                return output
            h2 = new.reshape(B, S, D)
        else:
            new = self.edit(h[:, self.last_idx, :])
            if new is None:
                return output
            h2 = h.clone()
            h2[:, self.last_idx, :] = new
        return (h2,) + output[1:] if is_tuple else h2


class AdditiveHook(SteeringHook):
    """h <- h + alpha * v_unit. Used by probe and contrastive.

    If norm_match=True, alpha is reinterpreted as a fractional coefficient k:
    the actual injected vector becomes (k * ||h_last||) * v_unit, so the
    perturbation has norm k * ||h_last||. This makes alpha layer/model agnostic.
    In broadcast mode, the same scaled vector is added to every token position
    (||h_last|| from the last token is the single scale used uniformly)."""

    def __init__(self, v_unit: torch.Tensor, alpha: float,
                 broadcast: bool = False, norm_match: bool = False):
        super().__init__(broadcast=broadcast)
        self.v_unit = v_unit
        self.alpha = alpha
        self.norm_match = norm_match

    def edit(self, h):
        # used only in last-token NON-norm-match path; norm_match path goes
        # through the overridden __call__ below.
        if self.alpha == 0.0:
            return None
        return h + (self.alpha * self.v_unit).to(h.dtype)

    def __call__(self, module, inputs, output):
        if not self.norm_match:
            return super().__call__(module, inputs, output)
        # norm-match path: scale magnitude by ||h_last||
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if self.last_idx is None or self.alpha == 0.0:
            return output
        h_last = h[:, self.last_idx, :]                              # [B, D]
        scale = (self.alpha * h_last.norm(dim=-1, keepdim=True))     # [B, 1]
        delta = (scale * self.v_unit.to(scale.dtype)).to(h.dtype)    # [B, D]
        if self.broadcast:
            h2 = h + delta.unsqueeze(1)                              # add to all tokens
        else:
            h2 = h.clone()
            h2[:, self.last_idx, :] = h_last + delta
        return (h2,) + output[1:] if is_tuple else h2


class AdditiveTwoHook(SteeringHook):
    """h <- h + alpha_V * v_V + alpha_T * v_T. Two-vector counterfactual steering.

    v_V and v_T are independent unit vectors (e.g., v_V_causal = mean_i[h_V - h_VT]
    over V-grounded items, v_T_causal = mean_i[h_T - h_VT] over T-grounded items).
    Each coefficient pushes toward its modality independently.

    Under norm_match=True, each coefficient is interpreted as a fractional
    coefficient k: the injected component along v_X has norm k_X * ||h_last||.
    Same broadcast / last-token routing as AdditiveHook."""

    def __init__(self, v_V_unit: torch.Tensor, v_T_unit: torch.Tensor,
                 alpha_V: float, alpha_T: float,
                 broadcast: bool = False, norm_match: bool = False):
        super().__init__(broadcast=broadcast)
        self.v_V = v_V_unit
        self.v_T = v_T_unit
        self.aV = alpha_V
        self.aT = alpha_T
        self.norm_match = norm_match

    def edit(self, h):
        if self.aV == 0.0 and self.aT == 0.0:
            return None
        return h + (self.aV * self.v_V + self.aT * self.v_T).to(h.dtype)

    def __call__(self, module, inputs, output):
        if not self.norm_match:
            return super().__call__(module, inputs, output)
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if self.last_idx is None or (self.aV == 0.0 and self.aT == 0.0):
            return output
        h_last = h[:, self.last_idx, :]
        h_norm = h_last.norm(dim=-1, keepdim=True)            # [B, 1]
        delta = ((self.aV * h_norm) * self.v_V.to(h_norm.dtype) +
                 (self.aT * h_norm) * self.v_T.to(h_norm.dtype)).to(h.dtype)  # [B, D]
        if self.broadcast:
            h2 = h + delta.unsqueeze(1)
        else:
            h2 = h.clone()
            h2[:, self.last_idx, :] = h_last + delta
        return (h2,) + output[1:] if is_tuple else h2


class SAEHook(SteeringHook):
    """Multiplicatively scale vision_top by (1+alpha) and text_top by (1-alpha)
    in SAE feature space, then add the decoded delta back to h_at_last.

    If norm_match=True, after computing the SAE-decoded delta we rescale it to
    have norm |alpha| * ||h_last||. This preserves the delta's direction (set
    by the latent edit and current SAE features) while pinning its magnitude
    to a fraction of the activation norm — so alpha becomes O(1) and
    comparable across layers, just like the additive NM case.

    If latent_additive=True, the per-atom edit becomes ADDITIVE instead of
    multiplicative:
        z'[v_idx] = z[v_idx] + alpha * v_target[v_idx]
        z'[t_idx] = z[t_idx] - alpha * t_target[t_idx]
    where v_target / t_target are per-atom magnitudes supplied at construction
    (e.g., rho_V_causal / rho_T_causal). This properly amplifies low-firing
    causal atoms that the multiplicative hook under-scales."""

    def __init__(self, sae, vision_idx: torch.Tensor, text_idx: torch.Tensor,
                 alpha: float, broadcast: bool = False, norm_match: bool = False,
                 latent_additive: bool = False,
                 v_target: torch.Tensor | None = None,
                 t_target: torch.Tensor | None = None):
        super().__init__(broadcast=broadcast)
        self.sae = sae
        self.v_idx = vision_idx
        self.t_idx = text_idx
        self.alpha = alpha
        self.norm_match = norm_match
        self.latent_additive = latent_additive
        # per-atom additive magnitudes (length == v_idx / t_idx); None if multiplicative
        self.v_target = v_target
        self.t_target = t_target
        self.delta_norms: list[float] = []
        self.zero_count: int = 0

    def edit(self, h_at_last):
        # only used in last-token NON-norm-match mode; other modes go through __call__
        if self.alpha == 0.0:
            return None
        delta = self._compute_delta(h_at_last)
        return h_at_last + delta.to(h_at_last.dtype)

    def _compute_delta(self, h_at_last: torch.Tensor) -> torch.Tensor:
        """Compute the SAE-mediated activation delta from the last-token slice.

        alpha lives in the SAE latent space: vision-top feats are scaled by
        (1+alpha), text-top feats by (1-alpha). The decoded difference is the
        delta that gets applied — either at the last position (default) or
        broadcast over every position (broadcast=True).

        Under norm_match, the decoded delta is rescaled to have norm
        |alpha| * ||h_last||, keeping its direction."""
        x = h_at_last.to(torch.float32)
        out = self.sae(x)
        z = out["z"]
        z2 = z.clone()
        if self.latent_additive:
            # z'[idx] = z[idx] + alpha * target[idx]
            if self.v_idx.numel() > 0 and self.v_target is not None:
                z2[:, self.v_idx] = z[:, self.v_idx] + self.alpha * self.v_target.to(z.dtype)
            if self.t_idx.numel() > 0 and self.t_target is not None:
                z2[:, self.t_idx] = z[:, self.t_idx] - self.alpha * self.t_target.to(z.dtype)
        else:
            # multiplicative (original): z'[idx] = z[idx] * (1±alpha)
            if self.v_idx.numel() > 0:
                z2[:, self.v_idx] = z[:, self.v_idx] * (1.0 + self.alpha)
            if self.t_idx.numel() > 0:
                z2[:, self.t_idx] = z[:, self.t_idx] * (1.0 - self.alpha)
        delta = self.sae.decode(z2) - self.sae.decode(z)
        if self.norm_match:
            # rescale delta to norm |alpha| * ||h_last||, per-row
            d_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h_norm = h_at_last.to(torch.float32).norm(dim=-1, keepdim=True)
            target = abs(self.alpha) * h_norm
            delta = delta * (target / d_norm)
        dn = float(delta.norm().item())
        self.delta_norms.append(dn)
        if dn == 0.0:
            self.zero_count += 1
        return delta

    def __call__(self, module, inputs, output):
        if not self.broadcast:
            return super().__call__(module, inputs, output)
        # broadcast mode: compute delta from last-token only, add to ALL positions
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if self.last_idx is None or self.alpha == 0.0:
            return output
        delta = self._compute_delta(h[:, self.last_idx, :])  # [B, D]
        h2 = h + delta.unsqueeze(1).to(h.dtype)              # broadcast over seq
        return (h2,) + output[1:] if is_tuple else h2


@torch.inference_mode()
def score_row(model, processor, row, hook, layer_idx, letter_token_ids):
    msgs = build_messages(row)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    last_idx = int(inputs.attention_mask.sum(dim=1).item() - 1)
    handle = None
    if hook is not None:
        hook.last_idx = last_idx
        layer = model.model.language_model.layers[layer_idx]
        handle = layer.register_forward_hook(hook)
    try:
        out = model(**inputs)
    finally:
        if handle is not None:
            handle.remove()
            hook.last_idx = None
    logits = out.logits[0, last_idx]
    return int(logits[letter_token_ids].argmax().item())


# Backbones whose chat template already renders a literal BOS token. The generic HF path builds a
# prompt with apply_chat_template(tokenize=False) and then tokenizes it in a separate processor
# call, which defaults to add_special_tokens=True -- so for these the sequence would get TWO BOS
# tokens. Callers pass add_special_tokens=False when the model_id is listed here.
#
# Keyed by model_id (every call site has it) and strictly OPT-IN: do NOT replace this with runtime
# auto-detection. Some already-evaluated backbones render a BOS-like marker in their template too,
# and flipping them would shift every number they have already produced. Adding a model here is a
# decision to re-run that model's artifacts, not a free correctness fix.
CHAT_TEMPLATE_EMITS_BOS = {
                           # 12B shares the 4b/27b chat template (renders a literal <bos>), so it
                           # double-BOSes under the library default exactly as they did. Registered

                           
                           # Measured 2026-08-22: template renders a literal <s> and the processor
                           # default adds a second (ids [1, 1, ...]); with add_special_tokens=False
                           # exactly one remains. Registered BEFORE any mistral24b artifact existed,
                           # so no produced number moves.
                           "mistralai/Mistral-Small-3.1-24B-Instruct-2503"}


def special_token_kwargs(model_id: str) -> dict:
    """Processor kwargs that keep exactly one BOS for `model_id`. Empty dict (=library default,
    byte-identical to the historical call) for every model not in CHAT_TEMPLATE_EMITS_BOS."""
    return {"add_special_tokens": False} if model_id in CHAT_TEMPLATE_EMITS_BOS else {}


def get_letter_token_ids(processor, device, n_letters: int = 4) -> torch.Tensor:
    """Token ids for the option letters A..(A+n_letters-1).

    Default n_letters=4 reproduces the prior A--D tensor exactly. For mixed-arity
    benchmark pools, request the max arity (e.g. 6) once and slice the first
    len(options) ids per row at decode time."""
    ids = []
    for L in LETTERS[:n_letters]:
        toks = processor.tokenizer.encode(L, add_special_tokens=False)
        if len(toks) != 1:
            toks = processor.tokenizer.encode(" " + L, add_special_tokens=False)
        ids.append(toks[0])
    return torch.tensor(ids, device=device)


def load_oracle_passed(data_root: Path, splits: tuple[str, ...] = ("train", "val", "test"),
                       dm_dir: Path | None = None) -> list[dict]:
    """All splits/{train,val,test}.jsonl items are oracle-passed by construction."""
    if dm_dir is None:
        dm_dir = data_root / "dm"
    rows = []
    for sub in splits:
        for line in (dm_dir / "splits" / f"{sub}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["__split"] = sub
            rows.append(r)
    return rows


def load_distraction_pool(path: Path) -> list[dict]:
    """Load distraction-event rows emitted by an extractor (single-modality
    correct + V+T wrong).

    Rows are tagged `__split = "distraction"` and carry `label = vision|text`
    (the direction that should help recovery). Caller appends these to
    all_rows; they flow through baseline + hooked phases identically to
    oracle survivors and `select_pool("unseen")` always includes them.

    Tagging as a new split (rather than "test") keeps the new test-held-out
    policy clean: distraction is its own first-class category, not test.
    """
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["__split"] = "distraction"   # override any prior __split tag
        rows.append(r)
    return rows


def baseline_rows_needed(pool_name: str) -> tuple[str, ...]:
    """Which splits we must baseline-score to determine pool membership.

    NOTE (2026-06): the "unseen" pool no longer includes test. Test is held
    out for the final evaluation; iteration on steering happens on
    val + model-fail-from-train.
    """
    if pool_name == "test":
        return ("test",)
    return ("train", "val")


def select_pool(all_rows: list[dict], baseline_preds: dict[str, int], pool_name: str) -> list[dict]:
    """Pool semantics:
      - "test"   : held-out test only (used for final eval, NOT iteration).
      - "all"    : every row loaded.
      - "unseen" : every val row (regardless of model V+T outcome)
                   PLUS every train row where model V+T was wrong
                   PLUS any distraction-pool rows (__split="distraction") if present.
                   Test is intentionally excluded.
      - "fails"  : every row where model V+T was wrong + distraction.
    """
    if pool_name == "test":
        return [r for r in all_rows if r["__split"] == "test"]
    if pool_name == "all":
        return list(all_rows)
    out = []
    for r in all_rows:
        cid = r["candidate_id"]
        if r["__split"] == "distraction":
            out.append(r); continue
        if cid not in baseline_preds:
            # baseline phase hasn't scored this row yet (mid-run resume); skip it
            continue
        is_val = r["__split"] == "val"
        is_fail = baseline_preds[cid] != r["correct_index"]
        if pool_name == "unseen" and (is_val or is_fail):
            out.append(r)
        elif pool_name == "fails" and is_fail:
            out.append(r)
    return out


def load_resume_state(out_path: Path) -> tuple[set, dict[str, int]]:
    done: set = set()
    baseline_preds: dict[str, int] = {}
    if not out_path.exists():
        return done, baseline_preds
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((r["candidate_id"], r["layer"], r["alpha"]))
        if r["layer"] == -1:
            baseline_preds[r["candidate_id"]] = r["pred"]
    return done, baseline_preds


def run_baseline_phase(*, model, processor, rows, done, out_f, baseline_preds,
                       letter_token_ids, score_fn=None):
    score_fn = score_fn or score_row
    todo = [r for r in rows if (r["candidate_id"], -1, 0.0) not in done]
    print(f"baseline phase: {len(todo)} forward passes")
    pbar = tqdm(total=len(todo), desc="baseline")
    for row in todo:
        pred = score_fn(model, processor, row, None, None, letter_token_ids)
        out_f.write(json.dumps({
            "candidate_id": row["candidate_id"], "label": row["label"],
            "correct_index": row["correct_index"], "split": row["__split"],
            "layer": -1, "alpha": 0.0, "pred": pred,
        }) + "\n")
        out_f.flush()
        baseline_preds[row["candidate_id"]] = pred
        pbar.update(1)
    pbar.close()


HookFactory = Callable[[int, float], SteeringHook]


def run_hooked_phase(*, model, processor, pool, layers, alphas, done, out_f,
                     hook_factory: HookFactory, letter_token_ids, desc="steer",
                     post_call: Callable[[int, float, SteeringHook], None] | None = None,
                     score_fn=None):
    score_fn = score_fn or score_row
    todo = []
    for row in pool:
        for L in layers:
            for a in alphas:
                if (row["candidate_id"], L, a) not in done:
                    todo.append((row, L, a))
    print(f"hooked phase: {len(todo)} forward passes")
    pbar = tqdm(total=len(todo), desc=desc)
    for row, L, a in todo:
        hook = hook_factory(L, a)
        pred = score_fn(model, processor, row, hook, L, letter_token_ids)
        if post_call is not None:
            post_call(L, a, hook)
        out_f.write(json.dumps({
            "candidate_id": row["candidate_id"], "label": row["label"],
            "correct_index": row["correct_index"], "split": row["__split"],
            "layer": L, "alpha": a, "pred": pred,
        }) + "\n")
        out_f.flush()
        pbar.update(1)
    pbar.close()


def _row_subpool(row: dict) -> str:
    """Map a pool row to its subpool tag for summary aggregation.

    - `oracle`        : original keep-rule survivor (no distraction_kind).
    - `v_distracted`  : Qwen V-only succeeded, V+T failed (caption distracted).
    - `t_distracted`  : Qwen T-only succeeded, V+T failed (image distracted).
    """
    k = row.get("distraction_kind")
    return k if k in ("v_distracted", "t_distracted") else "oracle"


def compute_summary(out_path: Path, pool: list[dict], layers, alphas) -> dict:
    pool_cids = {r["candidate_id"] for r in pool}
    rows_log = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    base = {r["candidate_id"]: r for r in rows_log
            if r["layer"] == -1 and r["candidate_id"] in pool_cids}
    if len(base) != len(pool):
        missing = len(pool) - len(base)
        print(f"  ! baseline coverage {len(base)}/{len(pool)} — dropping {missing} item(s) with no baseline pred")
        pool = [r for r in pool if r["candidate_id"] in base]
        pool_cids = {r["candidate_id"] for r in pool}

    item_meta = {}
    for row in pool:
        b = base[row["candidate_id"]]
        item_meta[row["candidate_id"]] = {
            "label": row["label"],
            "correct_index": row["correct_index"],
            "split": row["__split"],
            "baseline_pred": b["pred"],
            "baseline_correct": b["pred"] == row["correct_index"],
            "subpool": _row_subpool(row),
        }

    # `all` aggregate across subpools + per-subpool breakdown.
    subpools_present = sorted({m["subpool"] for m in item_meta.values()})
    has_dist = any(s != "oracle" for s in subpools_present)
    subpool_groups = ["all"] + (subpools_present if has_dist else [])

    summary: dict = {}
    for sub in subpool_groups:
        for label in ["vision", "text"]:
            items = [m for m in item_meta.values() if m["label"] == label
                     and (sub == "all" or m["subpool"] == sub)]
            n = len(items)
            n_pass = sum(1 for m in items if m["baseline_correct"])
            tag = label if sub == "all" else f"{sub}|{label}"
            summary[f"baseline|{tag}"] = {
                "n": n, "n_pass": n_pass, "n_fail": n - n_pass,
                "acc": n_pass / n if n else 0.0,
            }

    for L in layers:
        for a in alphas:
            for sub in subpool_groups:
                for label in ["vision", "text"]:
                    for status in ["pass", "fail"]:
                        cids = [cid for cid, m in item_meta.items()
                                if m["label"] == label
                                and (sub == "all" or m["subpool"] == sub)
                                and (m["baseline_correct"] if status == "pass" else not m["baseline_correct"])]
                        if not cids:
                            continue
                        cid_set = set(cids)
                        rs = [r for r in rows_log
                              if r["layer"] == L and r["alpha"] == a and r["candidate_id"] in cid_set]
                        if len(rs) != len(cids):
                            present = {r["candidate_id"] for r in rs}
                            cids = [c for c in cids if c in present]
                            if not cids:
                                continue
                        rs_correct = [r["pred"] == item_meta[r["candidate_id"]]["correct_index"] for r in rs]
                        rs_flip = [r["pred"] != item_meta[r["candidate_id"]]["baseline_pred"] for r in rs]
                        base_acc = float(np.mean([item_meta[c]["baseline_correct"] for c in cids]))
                        new_acc = float(np.mean(rs_correct))
                        tag = f"{label}|{status}" if sub == "all" else f"{sub}|{label}|{status}"
                        summary[f"L{L}|alpha={a}|{tag}"] = {
                            "n": len(cids),
                            "baseline_acc": base_acc,
                            "steered_acc": new_acc,
                            "delta_acc": new_acc - base_acc,
                            "flip_rate": float(np.mean(rs_flip)),
                            "subpool": sub,
                        }
    return summary


def print_summary(summary: dict, layers, alphas):
    # Detect which subpools are present (anything beyond the `all` aggregate).
    subpools = sorted({s["subpool"] for k, s in summary.items()
                       if k.startswith("L") and "subpool" in s and s["subpool"] != "all"})
    groups = ["all"] + subpools
    for L in layers:
        for a in alphas:
            header_printed = False
            for sub in groups:
                block_lines = []
                for label in ["vision", "text"]:
                    for status in ["pass", "fail"]:
                        tag = f"{label}|{status}" if sub == "all" else f"{sub}|{label}|{status}"
                        k = f"L{L}|alpha={a}|{tag}"
                        if k in summary:
                            s = summary[k]
                            block_lines.append(
                                f"  {label:>6}/{status:>4} (n={s['n']:3d}): "
                                f"acc {s['baseline_acc']:.3f} -> {s['steered_acc']:.3f} "
                                f"(Δ={s['delta_acc']:+.3f}, flip={s['flip_rate']:.3f})"
                            )
                if not block_lines:
                    continue
                if not header_printed:
                    print(f"\nL{L} alpha={a:+g}")
                    header_printed = True
                if sub != "all":
                    print(f" [{sub}]")
                for line in block_lines:
                    print(line)
