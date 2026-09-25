"""End-to-end oracle pipeline using an API-based oracle (default: Gemini 3 Flash).

Runs generation + 3 answer passes (V+T, V-only, T-only) and applies the
keep-rule on N seeds. Writes to a parallel directory so the original
sub-agent run under data/dm/ is untouched.

Outputs (under --out-dir, default data/dm/gemini/):
  candidates.jsonl              # post-shuffle, post-forbidden-filter
  work/gen/results.jsonl        # raw oracle generation outputs
  work/ans_vt/results.jsonl     # answer pass with image + caption
  work/ans_v/results.jsonl      # answer pass with image only
  work/ans_t/results.jsonl      # answer pass with caption only
  survivors.jsonl               # keep-rule survivors with pass_* flags
  summary.json                  # counts + keep rate
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from moground.oracle_clients import ImageInput, make_client
from moground.oracle_batch import submit_and_wait as _batch_submit_and_wait
from moground.oracle_prompts import (
    ANSWER_INSTRUCTIONS,
    GENERATION_INSTRUCTIONS,
    GENERATION_INSTRUCTIONS_DCI_HARD_T,
    GENERATION_INSTRUCTIONS_DCI_HARD_T_TONLY,
    GENERATION_INSTRUCTIONS_VISTEXT,
    GENERATION_INSTRUCTIONS_VISTEXT_HARD_T,
    GENERATION_INSTRUCTIONS_VISTEXT_HARD_T_TONLY,
    GENERATION_SYSTEM,
    validate_generation_output,
)

# Reuse the same T-block verifier as the subagent pipeline so both backends
# enforce identical C1 / bridge-structural constraints.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_pp02b", Path(__file__).parent / "postprocess_generation.py")
_pp02b = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_pp02b)  # type: ignore
verify_text_block = _pp02b.verify_text_block


FORBIDDEN_RE = re.compile(
    r"\b(image|picture|photo|photograph|caption|text|description|shown|"
    r"depicted|visible|pictured|displayed|according to|based on)\b",
    re.IGNORECASE,
)


def _strip_json(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip()


def _shuffle(options, correct_index, rng):
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(correct_index)


def _load_done(path: Path, key: str) -> set[str]:
    """Return ids that completed without error. Rows with an `error` field are
    treated as not-done so a resume re-attempts them."""
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                if key in r and "error" not in r:
                    done.add(r[key])
            except json.JSONDecodeError:
                pass
    return done


def _validate_tonly(obj: dict) -> list[str]:
    """T-only schema check: only the `text` block is required (no `vision`)."""
    errs: list[str] = []
    if "text" not in obj:
        errs.append("missing key 'text'")
        return errs
    sub = obj["text"]
    for k in ("question", "options", "correct_index"):
        if k not in sub:
            errs.append(f"text.{k} missing")
    if "options" in sub and not (isinstance(sub["options"], list) and len(sub["options"]) == 4):
        errs.append("text.options must be a list of 4")
    if "correct_index" in sub and sub["correct_index"] not in (0, 1, 2, 3):
        errs.append("text.correct_index must be 0..3")
    return errs


def _parse_gen_response(seed_id: str, text: str | None, error: str | None, t_only: bool) -> dict:
    if error:
        return {"seed_id": seed_id, "error": f"batch: {error}"}
    try:
        obj = json.loads(_strip_json(text or ""))
        errs = _validate_tonly(obj) if t_only else validate_generation_output(obj)
        if errs:
            return {"seed_id": seed_id, "error": "; ".join(errs)}
        return {"seed_id": seed_id, "result": obj}
    except Exception as e:
        return {"seed_id": seed_id, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def run_generation(client, seeds, out_path: Path, sys_prompt: str, *, max_workers: int,
                   t_only: bool, max_image_dim: int | None,
                   use_batch: bool = False, batch_poll_seconds: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path, "seed_id")
    todo = [s for s in seeds if s["seed_id"] not in done]
    if not todo:
        print(f"  generation: all {len(seeds)} done")
        return

    user_tmpl = (
        "seed_id: {sid}\noriginal_caption: {cap}\n\n"
        "Produce the JSON object defined in the instructions for the attached image."
    )

    if use_batch:
        requests = [{
            "key": s["seed_id"],
            "system": sys_prompt,
            "user": user_tmpl.format(sid=s["seed_id"], cap=s["caption"]),
            "images": [ImageInput(path=s["image_path"], max_dim=max_image_dim)],
            "temperature": 0.0,
            "response_mime_type": "application/json",
        } for s in todo]
        print(f"  batch gen: submitting {len(requests)} requests")
        results = _batch_submit_and_wait(
            gemini_client=client, requests=requests, work_dir=out_path.parent,
            display_name=f"gen-{out_path.parent.parent.name}",
            poll_seconds=batch_poll_seconds,
        )
        with out_path.open("a") as f:
            for s, r in zip(todo, results):
                row = _parse_gen_response(s["seed_id"], r.get("text"), r.get("error"), t_only)
                f.write(json.dumps(row) + "\n")
        return

    def call_one(seed):
        try:
            txt = client.complete(
                system=sys_prompt,
                user=user_tmpl.format(sid=seed["seed_id"], cap=seed["caption"]),
                images=[ImageInput(path=seed["image_path"], max_dim=max_image_dim)],
                temperature=0.0,
                response_mime_type="application/json",
            )
            obj = json.loads(_strip_json(txt))
            errs = _validate_tonly(obj) if t_only else validate_generation_output(obj)
            if errs:
                return {"seed_id": seed["seed_id"], "error": "; ".join(errs)}
            return {"seed_id": seed["seed_id"], "result": obj}
        except Exception as e:
            return {"seed_id": seed["seed_id"], "error": f"{type(e).__name__}: {str(e)[:200]}"}

    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(call_one, s) for s in todo]
        for fu in tqdm(as_completed(futures), total=len(futures), desc="gen"):
            row = fu.result()
            f.write(json.dumps(row) + "\n")
            f.flush()


def build_candidates(seeds_by_id, gen_path: Path, candidates_path: Path, seed: int,
                     run_c1_verifier: bool = True) -> int:
    n_in = n_err = n_out = n_forbidden = n_bridge_drop = 0
    bridge_drop_reasons: dict[str, int] = {}
    with candidates_path.open("w") as out:
        for line in gen_path.read_text().splitlines():
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            if "error" in row or "result" not in row:
                n_err += 1
                continue
            sid = row["seed_id"]
            seed_row = seeds_by_id.get(sid)
            if seed_row is None:
                n_err += 1
                continue
            res = row["result"]
            vq = res.get("vision", {}).get("question", "")
            tq = res.get("text", {}).get("question", "")
            if FORBIDDEN_RE.search(vq) or FORBIDDEN_RE.search(tq):
                n_forbidden += 1
                continue

            # C1/C2 + bridge structural check (shared verifier with 02b).
            # Only runs under HARD-T; the simple T-prompt embeds the answer
            # directly into the augmented caption by design (C1 would always fire).
            text_block = res.get("text")
            if text_block is not None and run_c1_verifier:
                reason = verify_text_block(text_block)
                if reason is not None:
                    n_bridge_drop += 1
                    bridge_drop_reasons[reason] = bridge_drop_reasons.get(reason, 0) + 1
                    text_block = None  # drop just the T-block; vision survives if valid

            labels = []
            if "vision" in res:
                labels.append(("vision", res["vision"], seed_row["caption"]))
            if text_block is not None:
                labels.append(("text", text_block,
                               text_block.get("augmented_caption", seed_row["caption"])))
            for label, block, cap in labels:
                rng = random.Random(f"{seed}|{sid}|{label}")
                opts, ci = _shuffle(block["options"], int(block["correct_index"]), rng)
                out.write(json.dumps({
                    "candidate_id": f"{sid}_{label[0]}",
                    "seed_id": sid,
                    "image_path": seed_row["image_path"],
                    "original_caption": seed_row["caption"],
                    "label": label,
                    "question": block["question"],
                    "options": opts,
                    "correct_index": ci,
                    "caption_for_filter": cap,
                    "rationale": block.get("rationale", ""),
                }) + "\n")
                n_out += 1
    print(f"  postprocess: read {n_in} ({n_err} errors, {n_forbidden} forbidden, "
          f"{n_bridge_drop} T-blocks dropped by C1/bridge verifier); wrote {n_out} candidates")
    if bridge_drop_reasons:
        print(f"    bridge_drop reasons: {bridge_drop_reasons}")
    return n_out


def _format_answer_user(c: dict, mode: str) -> str:
    parts = [f"candidate_id: {c['candidate_id']}"]
    if mode in ("vt", "t"):
        parts.append(f"caption: {c['caption_for_filter']}")
    parts.append(f"question: {c['question']}")
    parts.append("options:")
    for i, opt in enumerate(c["options"]):
        parts.append(f"  {i}. {opt}")
    parts.append("\nAnswer with a JSON object only.")
    return "\n".join(parts)


def _parse_answer_response(cid: str, text: str | None, error: str | None) -> dict:
    if error:
        return {"candidate_id": cid, "error": f"batch: {error}"}
    try:
        obj = json.loads(_strip_json(text or ""))
        pi = int(obj["predicted_index"])
        if pi not in (0, 1, 2, 3):
            return {"candidate_id": cid, "error": f"bad predicted_index: {pi}"}
        return {"candidate_id": cid, "predicted_index": pi}
    except Exception as e:
        return {"candidate_id": cid, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def run_answer_pass(client, candidates, out_path: Path, mode: str, *, max_workers: int,
                    max_image_dim: int | None = None,
                    use_batch: bool = False, batch_poll_seconds: int = 60) -> None:
    """mode in {'vt', 'v', 't'}: which modalities to provide."""
    assert mode in ("vt", "v", "t")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path, "candidate_id")
    todo = [c for c in candidates if c["candidate_id"] not in done]
    if not todo:
        print(f"  ans_{mode}: all {len(candidates)} done")
        return

    def _images_for(c):
        return [ImageInput(path=c["image_path"], max_dim=max_image_dim)] if mode in ("vt", "v") else []

    if use_batch:
        requests = [{
            "key": c["candidate_id"],
            "system": ANSWER_INSTRUCTIONS,
            "user": _format_answer_user(c, mode),
            "images": _images_for(c),
            "temperature": 0.0,
            "response_mime_type": "application/json",
        } for c in todo]
        print(f"  batch ans_{mode}: submitting {len(requests)} requests")
        results = _batch_submit_and_wait(
            gemini_client=client, requests=requests, work_dir=out_path.parent,
            display_name=f"ans_{mode}-{out_path.parent.parent.parent.name}",
            poll_seconds=batch_poll_seconds,
        )
        with out_path.open("a") as f:
            for c, r in zip(todo, results):
                row = _parse_answer_response(c["candidate_id"], r.get("text"), r.get("error"))
                f.write(json.dumps(row) + "\n")
        return

    def call_one(c):
        images = _images_for(c)
        try:
            txt = client.complete(
                system=ANSWER_INSTRUCTIONS,
                user=_format_answer_user(c, mode),
                images=images,
                temperature=0.0,
                response_mime_type="application/json",
            )
            obj = json.loads(_strip_json(txt))
            pi = int(obj["predicted_index"])
            if pi not in (0, 1, 2, 3):
                return {"candidate_id": c["candidate_id"], "error": f"bad predicted_index: {pi}"}
            return {"candidate_id": c["candidate_id"], "predicted_index": pi}
        except Exception as e:
            return {"candidate_id": c["candidate_id"], "error": f"{type(e).__name__}: {str(e)[:200]}"}

    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(call_one, c) for c in todo]
        for fu in tqdm(as_completed(futures), total=len(futures), desc=f"ans_{mode}"):
            row = fu.result()
            f.write(json.dumps(row) + "\n")
            f.flush()


def load_predictions(path: Path) -> dict[str, int]:
    preds: dict[str, int] = {}
    if not path.exists():
        return preds
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "predicted_index" in r:
            preds[r["candidate_id"]] = r["predicted_index"]
    return preds


def apply_keep_rule(candidates, vt, v, t, out_path: Path,
                    *, skip_vt: bool = False) -> tuple[int, int, int, int]:
    """Legacy oracle keep rule (no Qwen). 3-pass by default; 2-pass when skip_vt.

    Keep rule:
      vision-grounded: (V+T correct AND) V-only correct AND T-only WRONG
      text-grounded:   (V+T correct AND) V-only WRONG    AND T-only correct
    The V+T gate is omitted when skip_vt=True (saves ~50% Gemini spend).
    """
    survivors = []
    skipped = 0
    for c in candidates:
        cid = c["candidate_id"]
        if cid not in v or cid not in t:
            skipped += 1
            continue
        if not skip_vt and cid not in vt:
            skipped += 1
            continue
        ci = c["correct_index"]
        v_ok, t_ok = v[cid] == ci, t[cid] == ci
        vt_ok = True if skip_vt else (vt[cid] == ci)
        if c["label"] == "vision" and vt_ok and v_ok and not t_ok:
            survivors.append({**c, "pass_vt": None if skip_vt else True,
                              "pass_v": True, "pass_t": False})
        elif c["label"] == "text" and vt_ok and not v_ok and t_ok:
            survivors.append({**c, "pass_vt": None if skip_vt else True,
                              "pass_v": False, "pass_t": True})
    with out_path.open("w") as f:
        for r in survivors:
            f.write(json.dumps(r) + "\n")
    n_v = sum(1 for r in survivors if r["label"] == "vision")
    n_t = sum(1 for r in survivors if r["label"] == "text")
    return len(survivors), n_v, n_t, skipped


def apply_oracle_cascade(candidates, t_preds, vt_preds, out_path: Path,
                         *, skip_vt: bool = False) -> tuple[int, int, int, int]:
    """Per-label oracle filter after T-only and (optionally) V+T passes.

    Step A (T-only):
      T-grounded: keep iff oracle T-only correct (caption sufficient).
      V-grounded: keep iff oracle T-only WRONG    (caption did not suffice → real V).
    Step B (V+T) — when skip_vt=False:
      Both labels: keep iff oracle V+T correct (well-posed question).
    When skip_vt=True: Step B is omitted. Rationale: malformed questions
    typically fail oracle T-only too, so the T-only gate already drops them;
    Qwen V-only adds further filtering downstream. Saves ~50% Gemini spend.

    Survivors are written to out_path (qwen_candidates.jsonl) with `pass_t`,
    `pass_vt` flags (`pass_vt` = None when skip_vt).
    """
    survivors = []
    skipped = 0
    for c in candidates:
        cid = c["candidate_id"]
        if cid not in t_preds:
            skipped += 1
            continue
        if not skip_vt and cid not in vt_preds:
            skipped += 1
            continue
        ci = c["correct_index"]
        t_ok = t_preds[cid] == ci
        if c["label"] == "text" and not t_ok:
            continue
        if c["label"] == "vision" and t_ok:
            continue
        if not skip_vt:
            vt_ok = vt_preds[cid] == ci
            if not vt_ok:
                continue
            survivors.append({**c, "pass_t": t_ok, "pass_vt": True})
        else:
            survivors.append({**c, "pass_t": t_ok, "pass_vt": None})
    with out_path.open("w") as f:
        for r in survivors:
            f.write(json.dumps(r) + "\n")
    n_v = sum(1 for r in survivors if r["label"] == "vision")
    n_t = sum(1 for r in survivors if r["label"] == "text")
    return len(survivors), n_v, n_t, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--num-seeds", type=int, default=100)
    ap.add_argument("--out-dir", default="data/dm/gemini")
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--key-path", default=".credentials/google_gladia")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-image-dim", type=int, default=None,
                    help="if set, longest image side is resized to this many pixels "
                         "before encoding (applies to gen + ans_vt + ans_v). "
                         "Recommended: 1024 — keeps signage readable, ~2-3x token savings.")
    ap.add_argument("--skip-vt-filter", action="store_true",
                    help="Skip the oracle V+T pass (run only ans_v + ans_t). "
                         "Rationale: malformed questions typically also fail oracle "
                         "T-only or V-only, so V+T is mostly redundant.")
    ap.add_argument("--use-batch", action="store_true",
                    help="Use Gemini Batch API for gen + answer passes "
                         "(50%% discount, async with 24h SLA). State is persisted "
                         "to work/<stage>/batch.txt for resume.")
    ap.add_argument("--batch-poll-seconds", type=int, default=60,
                    help="how often to poll the batch job state (default 60s)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    seed = int(cfg["seed"])

    # Pick the generation prompt the same way prepare_oracle_generation.py does,
    # so the Gemini pipeline matches the subagent pipeline byte-for-byte on prompt.
    hard_t = bool(cfg.get("seeds", {}).get("hard_t", True))
    t_only = bool(cfg.get("seeds", {}).get("t_only", False))
    if t_only and not hard_t:
        raise ValueError("seeds.t_only requires seeds.hard_t to also be True")
    src = cfg.get("seeds", {}).get("source", "dci")
    if src == "vistext":
        if t_only:    instructions = GENERATION_INSTRUCTIONS_VISTEXT_HARD_T_TONLY
        elif hard_t:  instructions = GENERATION_INSTRUCTIONS_VISTEXT_HARD_T
        else:         instructions = GENERATION_INSTRUCTIONS_VISTEXT
    else:
        if t_only:    instructions = GENERATION_INSTRUCTIONS_DCI_HARD_T_TONLY
        elif hard_t:  instructions = GENERATION_INSTRUCTIONS_DCI_HARD_T
        else:         instructions = GENERATION_INSTRUCTIONS
    # T-only preambles carry their own header; V+T mode prepends GENERATION_SYSTEM.
    sys_prompt = instructions if t_only else (GENERATION_SYSTEM + "\n\n" + instructions)
    print(f"prompt: source={src} hard_t={hard_t} t_only={t_only}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds_all = [json.loads(l) for l in (dm_dir / "seeds.jsonl").read_text().splitlines() if l.strip()]
    seeds = seeds_all[: args.num_seeds]
    seeds_by_id = {s["seed_id"]: s for s in seeds_all}
    print(f"using {len(seeds)} seeds; out_dir={out_dir}")

    client = make_client(args.backend, model=args.model, key_path=args.key_path)
    print(f"oracle: {args.backend}/{args.model}")

    gen_path = out_dir / "work" / "gen" / "results.jsonl"
    print("\n[1/4] generation" + (" [BATCH]" if args.use_batch else ""))
    run_generation(client, seeds, gen_path, sys_prompt,
                   max_workers=args.max_workers, t_only=t_only,
                   max_image_dim=args.max_image_dim,
                   use_batch=args.use_batch, batch_poll_seconds=args.batch_poll_seconds)

    candidates_path = out_dir / "candidates.jsonl"
    print("\n[2/4] postprocess generation -> candidates.jsonl")
    n_candidates = build_candidates(seeds_by_id, gen_path, candidates_path, seed,
                                    run_c1_verifier=hard_t)
    candidates = [json.loads(l) for l in candidates_path.read_text().splitlines() if l.strip()]

    # Full 3-pass oracle keep rule: V+T, V-only, T-only all judged by the oracle.
    modes = ["v", "t"] if args.skip_vt_filter else ["vt", "v", "t"]
    n_stages = 3 + len(modes)  # gen + postprocess + N answer passes + keep-rule
    print(f"\n[3/{n_stages}-{2+len(modes)}] oracle answer passes ({', '.join(modes)})"
          + (" (V+T SKIPPED)" if args.skip_vt_filter else ""))
    for mode in modes:
        run_answer_pass(
            client, candidates, out_dir / "work" / f"ans_{mode}" / "results.jsonl",
            mode, max_workers=args.max_workers, max_image_dim=args.max_image_dim,
            use_batch=args.use_batch, batch_poll_seconds=args.batch_poll_seconds,
        )
    print(f"\n[{n_stages}/{n_stages}] keep-rule ("
          + ("2-pass" if args.skip_vt_filter else "3-pass") + ")")
    vt = {} if args.skip_vt_filter else load_predictions(out_dir / "work" / "ans_vt" / "results.jsonl")
    v  = load_predictions(out_dir / "work" / "ans_v"  / "results.jsonl")
    t  = load_predictions(out_dir / "work" / "ans_t"  / "results.jsonl")
    n_surv, n_v, n_t, n_skip = apply_keep_rule(
        candidates, vt, v, t, out_dir / "survivors.jsonl",
        skip_vt=args.skip_vt_filter)
    summary = {
        "mode": ("oracle_2pass_no_vt" if args.skip_vt_filter else "oracle_3pass"),
        "n_seeds": len(seeds),
        "n_candidates": n_candidates,
        "n_survivors": n_surv,
        "n_vision_survivors": n_v,
        "n_text_survivors": n_t,
        "n_skipped_missing_preds": n_skip,
        "keep_rate": n_surv / n_candidates if n_candidates else 0.0,
        "model": args.model,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nseeds={len(seeds)}  candidates={n_candidates}  survivors={n_surv} "
          f"({n_v}v + {n_t}t)  keep_rate={summary['keep_rate']:.3f}")
    print(f"-> {out_dir / 'survivors.jsonl'}")


if __name__ == "__main__":
    main()
