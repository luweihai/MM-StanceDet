"""Thin client for calling a multimodal chat-completions endpoint.

The endpoint is expected to be OpenAI-compatible so that image content can be
passed as a base64 data URL. Retries are applied on transient failures.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import time
from pathlib import Path
from typing import List, Optional

import requests

from .config import LLMConfig


logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, cfg: LLMConfig, api_key: Optional[str] = None):
        self.cfg = cfg
        self.api_key = api_key

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _image_data_uri(path: str) -> str:
        p = Path(path)
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _build_messages(
        self, system: str, user_text: str, image_path: Optional[str]
    ) -> List[dict]:
        if image_path and Path(image_path).exists():
            content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_uri(image_path)},
                },
            ]
        else:
            content = user_text
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def complete(
        self,
        system: str,
        user_text: str,
        image_path: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a single prompt and return the assistant's text reply."""
        payload = {
            "model": self.cfg.model,
            "messages": self._build_messages(system, user_text, image_path),
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.post(
                    self.cfg.base_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.cfg.timeout_sec,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - retry covers network/HTTP errors
                last_err = exc
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

