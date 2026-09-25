"""Modular oracle-API client interface.

A backend implements `OracleClient.complete(system, user, images)` and returns
the raw text response. Add a new provider by subclassing and registering it
in `make_client`.

Currently implemented:
- gemini (google-genai SDK)

Stubs to add when needed:
- openai (gpt-* via openai SDK)
- anthropic (claude-* via anthropic SDK)
- local (HF transformers / vLLM / Ollama)
"""
from __future__ import annotations

import abc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class ImageInput:
    path: str | Path
    mime_type: str = "image/jpeg"
    max_dim: int | None = None  # if set, longest side is resized to this; aspect preserved


def _read_image_bytes(img: "ImageInput") -> tuple[bytes, str]:
    """Read image bytes from disk, optionally downsampling longest side to img.max_dim.

    Returns (bytes, mime_type). When resizing, always re-encodes as JPEG quality 90
    (smaller than PNG, no perceptual loss at these targets).
    """
    if img.max_dim is None:
        return Path(img.path).read_bytes(), img.mime_type
    from io import BytesIO
    try:
        from PIL import Image  # lazy import — only needed when resizing
    except ImportError as e:
        raise RuntimeError("--max-image-dim requires Pillow: pip install Pillow") from e
    im = Image.open(img.path)
    w, h = im.size
    longest = max(w, h)
    if longest <= img.max_dim:
        return Path(img.path).read_bytes(), img.mime_type
    scale = img.max_dim / longest
    im = im.convert("RGB").resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


class OracleClient(abc.ABC):
    """Provider-agnostic single-turn oracle interface."""

    name: str

    @abc.abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        images: Sequence[ImageInput] = (),
        *,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> str:
        ...


def _read_key(api_key: str | None, env_var: str, key_path: str | Path | None) -> str:
    if api_key:
        return api_key
    if key_path is not None:
        return Path(key_path).read_text().strip()
    val = os.environ.get(env_var)
    if not val:
        raise RuntimeError(f"no API key: pass api_key, key_path, or set ${env_var}")
    return val


class GeminiClient(OracleClient):
    """Google Gemini via google-genai SDK."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        *,
        api_key: str | None = None,
        key_path: str | Path | None = None,
        max_retries: int = 3,
    ):
        from google import genai
        self._genai = genai
        self._types = __import__("google.genai.types", fromlist=["types"])
        self.model = model
        self.client = genai.Client(api_key=_read_key(api_key, "GOOGLE_API_KEY", key_path))
        self.max_retries = max_retries

    def complete(self, system, user, images=(), *, temperature=0.0, max_output_tokens=None,
                 response_mime_type: str | None = None):
        types = self._types
        parts: list = []
        for img in images:
            data, mime = _read_image_bytes(img)
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=data)))
        parts.append(types.Part(text=user))

        cfg_kwargs = {"system_instruction": system, "temperature": temperature}
        if max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = max_output_tokens
        if response_mime_type is not None:
            cfg_kwargs["response_mime_type"] = response_mime_type
        config = types.GenerateContentConfig(**cfg_kwargs)

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=parts, config=config,
                )
                return resp.text or ""
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"gemini call failed after {self.max_retries} retries") from last_err


def make_client(backend: str, **kwargs) -> OracleClient:
    """Factory. backend in {'gemini'}. Extend by registering new classes here."""
    backends = {"gemini": GeminiClient}
    if backend not in backends:
        raise ValueError(f"unknown backend {backend!r}; available: {sorted(backends)}")
    return backends[backend](**kwargs)
