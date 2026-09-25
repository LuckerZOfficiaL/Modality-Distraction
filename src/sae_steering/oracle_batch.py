"""Gemini Batch API helpers.

Submit a list of (key, request_dict) tuples as a single batch job to Gemini's
file-based batch endpoint, poll for completion, and collect responses keyed
back to the inputs.

Batch API gives 50% discount vs sync, with a 24h SLA (often minutes-to-hours).

Usage:
    client = GeminiClient(model="gemini-3-flash-preview", key_path=...)
    out_jsonl = submit_and_wait(
        gemini_client=client,
        requests=[{"key": "seed_0001", "system": "...", "user": "...",
                   "images": [ImageInput(path=..., max_dim=1024)],
                   "temperature": 0.0,
                   "response_mime_type": "application/json"}, ...],
        work_dir=Path("data/dm/.../work/gen"),
        display_name="my-job-2026-05-26",
        poll_seconds=60,
    )
    # out_jsonl is a list[dict] with {"key": ..., "text": ..., "error": ...}

State files written to work_dir:
    batch_input.jsonl   — uploaded request file
    batch.txt           — Gemini batch job name (for resume)
    batch_output.jsonl  — downloaded response file (cached)
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Sequence

from .oracle_clients import GeminiClient, ImageInput, _read_image_bytes


def _build_request_object(req: dict, model: str, types) -> dict:
    """Translate our request dict → Gemini batch request JSON (one line)."""
    parts: list[dict] = []
    if req.get("parts") is not None:
        # explicit ordered parts: ("text", str) | ("image", ImageInput). Use this when the
        # interleaving matters, e.g. to reproduce a prompt where the caption precedes the image.
        for kind, val in req["parts"]:
            if kind == "image":
                data, mime = _read_image_bytes(val)
                parts.append({"inline_data": {"mime_type": mime,
                                              "data": base64.b64encode(data).decode()}})
            else:
                parts.append({"text": val})
    else:
        for img in req.get("images", ()):
            data, mime = _read_image_bytes(img)
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}})
        parts.append({"text": req["user"]})

    generation_config: dict = {}
    if req.get("temperature") is not None:
        generation_config["temperature"] = req["temperature"]
    if req.get("max_output_tokens") is not None:
        generation_config["maxOutputTokens"] = req["max_output_tokens"]
    if req.get("response_mime_type") is not None:
        generation_config["responseMimeType"] = req["response_mime_type"]
    if req.get("response_schema") is not None:
        # with an enum schema the model must emit one of the listed strings, which turns a free
        # generation into a forced choice and removes unparseable outputs by construction
        generation_config["responseSchema"] = req["response_schema"]
    if req.get("thinking_budget") is not None:
        generation_config["thinkingConfig"] = {"thinkingBudget": req["thinking_budget"]}

    request_body: dict = {
        "contents": [{"role": "user", "parts": parts}],
    }
    if req.get("system"):
        request_body["systemInstruction"] = {"parts": [{"text": req["system"]}]}
    if generation_config:
        request_body["generationConfig"] = generation_config

    return {"key": req["key"], "request": request_body}


def submit_and_wait(
    *,
    gemini_client: GeminiClient,
    requests: Sequence[dict],
    work_dir: Path,
    display_name: str,
    poll_seconds: int = 60,
) -> list[dict]:
    """Submit `requests` as one batch job; return ordered list aligned to inputs.

    Each returned item: {"key": ..., "text": <str|None>, "error": <str|None>}.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    client = gemini_client.client
    types = gemini_client._types
    model = gemini_client.model

    in_path = work_dir / "batch_input.jsonl"
    name_path = work_dir / "batch.txt"
    out_path = work_dir / "batch_output.jsonl"

    # 1. Build request file if not already built.
    if not in_path.exists():
        with in_path.open("w") as f:
            for req in requests:
                f.write(json.dumps(_build_request_object(req, model, types)) + "\n")
        print(f"  built {len(requests)} requests -> {in_path} ({in_path.stat().st_size/1e6:.1f} MB)")

    # 2. Submit or resume.
    if name_path.exists():
        batch_name = name_path.read_text().strip()
        print(f"  resuming batch: {batch_name}")
    else:
        print(f"  uploading {in_path.name} to Files API...")
        uploaded = client.files.upload(
            file=str(in_path),
            config={"display_name": f"{display_name}-input", "mime_type": "application/jsonl"},
        )
        print(f"  file: {uploaded.name}")
        print(f"  creating batch job model={model}")
        batch = client.batches.create(
            model=model,
            src=uploaded.name,
            config={"display_name": display_name},
        )
        batch_name = batch.name
        name_path.write_text(batch_name)
        print(f"  batch submitted: {batch_name}")

    # 3. Poll until done. Per docs, batch.state.name is an enum string;
    # terminal states are SUCCEEDED / FAILED / CANCELLED / EXPIRED.
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                       "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while True:
        batch = client.batches.get(name=batch_name)
        state = batch.state.name if hasattr(batch.state, "name") else str(batch.state)
        print(f"  [{time.strftime('%H:%M:%S')}] state={state}")
        if state in terminal_states:
            break
        time.sleep(poll_seconds)

    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"batch {batch_name} ended in state {state}: "
                           f"{getattr(batch, 'error', None)}")

    # 4. Download results.
    dest = batch.dest
    if dest is None:
        raise RuntimeError(f"batch {batch_name} has no destination set")
    out_file_name = dest.file_name
    if not out_file_name:
        raise RuntimeError(f"batch {batch_name} destination has no file_name: {dest}")

    if not out_path.exists():
        print(f"  downloading results from {out_file_name}")
        blob = client.files.download(file=out_file_name)
        out_path.write_bytes(blob)
        print(f"  -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    # 5. Parse responses, keyed.
    by_key: dict[str, dict] = {}
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key")
        if key is None:
            continue
        if "response" in row:
            text = ""
            try:
                resp = row["response"]
                # Extract text from the first candidate's first text part
                cands = resp.get("candidates", [])
                if cands:
                    parts = cands[0].get("content", {}).get("parts", [])
                    for p in parts:
                        if "text" in p:
                            text += p["text"]
            except Exception as e:
                by_key[key] = {"key": key, "text": None, "error": f"parse: {e}"}
                continue
            by_key[key] = {"key": key, "text": text, "error": None}
        elif "error" in row:
            by_key[key] = {"key": key, "text": None, "error": json.dumps(row["error"])[:300]}
        else:
            by_key[key] = {"key": key, "text": None, "error": "missing response and error"}

    return [by_key.get(r["key"], {"key": r["key"], "text": None, "error": "no response"})
            for r in requests]
