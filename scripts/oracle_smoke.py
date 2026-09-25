"""Smoke-test the modular oracle client with image+text input.

Usage:
  python scripts/oracle_smoke.py
  python scripts/oracle_smoke.py --backend gemini --model gemini-3-flash-preview
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sae_steering.oracle_clients import ImageInput, make_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--key-path", default=".credentials/google_gladia")
    ap.add_argument("--image", default="data/raw/coco/val2017/000000173830.jpg")
    ap.add_argument("--prompt", default="Describe this image in one sentence.")
    args = ap.parse_args()

    client = make_client(args.backend, model=args.model, key_path=args.key_path)
    img = Path(args.image)
    if not img.is_absolute():
        img = Path(__file__).resolve().parent.parent / img
    out = client.complete(
        system="You are a concise image describer.",
        user=args.prompt,
        images=[ImageInput(path=img)],
    )
    print(f"[{client.name}/{args.model}] {out}")


if __name__ == "__main__":
    main()
