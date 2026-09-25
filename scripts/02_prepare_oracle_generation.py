"""Step 2 (prep): split seeds into chunks of work orders for oracle agents.

Emits:
  data/dm/work/gen/chunk_{i:03d}.jsonl   # one row per seed, fields: seed_id, image_path, caption
  data/dm/work/gen/chunk_{i:03d}.result.jsonl  # empty; agents write here
  data/dm/work/gen/manifest.json         # list of chunks + agent prompt

Agents are spawned separately (from the Claude Code conversation) with one agent
per chunk, using the manifest's agent_prompt_template to tell each agent what
to produce.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sae_steering.oracle_prompts import (
    GENERATION_INSTRUCTIONS,
    GENERATION_INSTRUCTIONS_DCI_HARD_T,
    GENERATION_INSTRUCTIONS_DCI_HARD_T_TONLY,
    GENERATION_INSTRUCTIONS_VISTEXT,
    GENERATION_INSTRUCTIONS_VISTEXT_HARD_T,
    GENERATION_INSTRUCTIONS_VISTEXT_HARD_T_TONLY,
    GENERATION_SYSTEM,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--num-seeds", type=int, default=500,
                    help="use only the first N seeds from seeds.jsonl")
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--per-seed", action="store_true",
                    help="emit one input/output JSON file per seed (under work/gen/seeds/); "
                         "agent uses the Write tool — sidesteps shell quoting bugs and "
                         "subagent Bash permission denials. Recommended for all new runs.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dm_dir = Path(cfg["paths"]["dm_dir"])
    seeds_file = dm_dir / "seeds.jsonl"
    work_dir = dm_dir / "work" / "gen"
    work_dir.mkdir(parents=True, exist_ok=True)

    seeds = [json.loads(l) for l in seeds_file.read_text().splitlines() if l.strip()]
    seeds = seeds[: args.num_seeds]

    sources = {s.get("source", "dci") for s in seeds}
    hard_t = bool(cfg.get("seeds", {}).get("hard_t", True))
    t_only = bool(cfg.get("seeds", {}).get("t_only", False))
    if t_only and not hard_t:
        raise ValueError("seeds.t_only requires seeds.hard_t to also be True")
    if sources == {"vistext"}:
        if t_only:
            instructions = GENERATION_INSTRUCTIONS_VISTEXT_HARD_T_TONLY
        elif hard_t:
            instructions = GENERATION_INSTRUCTIONS_VISTEXT_HARD_T
        else:
            instructions = GENERATION_INSTRUCTIONS_VISTEXT
        tag = " HARD-T T-ONLY" if t_only else (" HARD-T" if hard_t else "")
        print(f"using VISTEXT generation instructions (chart-aware){tag}")
    else:
        if t_only:
            instructions = GENERATION_INSTRUCTIONS_DCI_HARD_T_TONLY
        elif hard_t:
            instructions = GENERATION_INSTRUCTIONS_DCI_HARD_T
        else:
            instructions = GENERATION_INSTRUCTIONS
        tag = " HARD-T T-ONLY" if t_only else (" HARD-T" if hard_t else "")
        print(f"using DEFAULT (DCI) generation instructions{tag}; sources={sources}")

    chunks: list[dict] = []
    seeds_subdir = work_dir / "seeds"
    if args.per_seed:
        seeds_subdir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            in_path = seeds_subdir / f"{s['seed_id']}.in.json"
            in_path.write_text(json.dumps({
                "seed_id": s["seed_id"],
                "image_path": s["image_path"],
                "caption": s["caption"],
            }))

    for i in range(0, len(seeds), args.chunk_size):
        batch = seeds[i : i + args.chunk_size]
        idx = i // args.chunk_size
        if args.per_seed:
            # Chunk-level manifest entry just lists seed_ids; agent writes one .out.json per seed.
            chunks.append({
                "index": idx,
                "seed_ids": [s["seed_id"] for s in batch],
                "seeds_dir": str(seeds_subdir),
                "n_items": len(batch),
            })
        else:
            in_path = work_dir / f"chunk_{idx:03d}.jsonl"
            out_path = work_dir / f"chunk_{idx:03d}.result.jsonl"
            with in_path.open("w") as f:
                for s in batch:
                    f.write(json.dumps({
                        "seed_id": s["seed_id"],
                        "image_path": s["image_path"],
                        "caption": s["caption"],
                    }) + "\n")
            out_path.touch(exist_ok=True)
            chunks.append({
                "index": idx,
                "input": str(in_path),
                "output": str(out_path),
                "n_items": len(batch),
            })

    # T-only instructions carry their own preamble; the V+T GENERATION_SYSTEM
    # ("produce TWO ... MCQs") would contradict it.
    system_prefix = "" if t_only else (GENERATION_SYSTEM + "\n\n")
    if args.per_seed:
        agent_prompt_template = (
            system_prefix
            + instructions
            + "\n\n"
            + "Your task (RESUMABLE, per-seed file mode):\n"
            + "1. You are given a list of seed_ids in {SEED_IDS} and a directory {SEEDS_DIR}.\n"
            + "2. For each seed_id in the list, the input is at {SEEDS_DIR}/<seed_id>.in.json\n"
            + "   (fields: seed_id, image_path, caption) and your output goes to\n"
            + "   {SEEDS_DIR}/<seed_id>.out.json.\n"
            + "3. RESUME: if {SEEDS_DIR}/<seed_id>.out.json already exists, SKIP that seed.\n"
            + "4. For each remaining seed_id:\n"
            + "   - Read {SEEDS_DIR}/<seed_id>.in.json to get image_path and caption.\n"
            + "   - Read the image at image_path with the Read tool.\n"
            + "   - Produce the JSON object described above (with seed_id included as a top-level field).\n"
            + "   - Write it to {SEEDS_DIR}/<seed_id>.out.json with the Write tool.\n"
            + "     The Write tool sidesteps shell-quoting bugs (apostrophes, embedded JSON) entirely.\n"
            + "5. If a particular image cannot be read, Write the file with content\n"
            + "     {\"seed_id\": \"...\", \"error\": \"<reason>\"}\n"
            + "   and continue. Error files count as done for resume purposes.\n"
            + "6. Stop when every seed_id in {SEED_IDS} has a corresponding .out.json file.\n"
            + "7. Do not write anything outside {SEEDS_DIR}/<seed_id>.out.json. Do not modify .in.json files.\n"
            + "8. Total target: {N_ITEMS} output files.\n"
        )
    else:
        agent_prompt_template = (
            system_prefix
            + instructions
            + "\n\n"
            + "Your task (RESUMABLE):\n"
            + "1. Read the input file at {INPUT_PATH} (one JSON object per line, fields: seed_id, image_path, caption).\n"
            + "2. Read the output file at {OUTPUT_PATH} if it exists. Build the set of seed_ids ALREADY PRESENT there (one JSON per line). These are already done and MUST be skipped.\n"
            + "3. For EACH input row whose seed_id is NOT yet in the output file:\n"
            + "   - Read the image at image_path (use the Read tool).\n"
            + "   - Produce the JSON object described above.\n"
            + "   - APPEND one line to {OUTPUT_PATH} using a Bash command like:\n"
            + "       python3 -c 'import json,sys; open(sys.argv[1],\"a\").write(sys.argv[2]+\"\\n\")' {OUTPUT_PATH} '<compact-json-of-row>'\n"
            + "     Do NOT use the Write tool (it overwrites). Each row must be appended ATOMICALLY and flushed to disk before the next image is read, so that interruption at any point leaves a valid JSONL with every completed row persisted.\n"
            + "4. If a particular image cannot be read or the task cannot be completed for a row, append\n"
            + "     {\"seed_id\": \"...\", \"error\": \"<reason>\"}\n"
            + "   and continue. Error rows count as done for resume purposes.\n"
            + "5. Stop when every input seed_id appears in the output file.\n"
            + "6. Do not write anything to disk outside {OUTPUT_PATH}. Do not modify the input file. Never rewrite or truncate {OUTPUT_PATH} \u2014 only append.\n"
            + "7. Total target: {N_ITEMS} output rows.\n"
        )

    manifest = {
        "mode": "per_seed" if args.per_seed else "chunk",
        "chunks": chunks,
        "agent_prompt_template": agent_prompt_template,
        "n_seeds": len(seeds),
        "n_chunks": len(chunks),
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {len(chunks)} chunks covering {len(seeds)} seeds -> {work_dir}")


if __name__ == "__main__":
    main()
