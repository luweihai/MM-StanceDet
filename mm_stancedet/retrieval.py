"""Retrieval-augmentation stage: encode samples and retrieve nearest neighbours.

The production encoder uses CLIP so that text and image are embedded into a
shared space. A lightweight hash-based encoder is kept for dependency-free
smoke tests of the surrounding pipeline.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable, List, Optional

import numpy as np

from .config import Config
from .schema import StanceSample


logger = logging.getLogger(__name__)


class ClipEncoder:
    """CLIP-based text/image encoder using the HF transformers stack."""

    def __init__(self, model_name: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with self.torch.no_grad():
            vec = self.model.get_text_features(**inputs)
        return vec[0].cpu().numpy()

    def encode_image(self, image_path: str) -> np.ndarray:
        from PIL import Image

        pil = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=pil, return_tensors="pt")
        with self.torch.no_grad():
            vec = self.model.get_image_features(**inputs)
        return vec[0].cpu().numpy()

    def encode_sample(self, sample: StanceSample, image_path_abs: Optional[str] = None) -> np.ndarray:
        t = self.encode_text(sample.text)
        i = self.encode_image(image_path_abs or sample.image_path)
        fused = (t + i) / 2.0
        return fused / (np.linalg.norm(fused) + 1e-8)


class SummaryEncoder:
    """Deterministic feature-hash encoder used for offline pipeline tests."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _tokens(text: str):
        return [tok for tok in text.lower().replace("\n", " ").split() if tok]

    def encode_sample(self, sample: StanceSample, image_path_abs: Optional[str] = None) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(sample.text)[:128]:
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)


class Retriever:
    """Vector store with cosine-similarity nearest-neighbour search."""

    def __init__(
        self,
        cfg: Config,
        encoder: Optional[object] = None,
        image_resolver: Optional[Callable[[StanceSample], str]] = None,
    ):
        self.cfg = cfg
        self._image_resolver = image_resolver or (lambda s: s.image_path)
        if encoder is not None:
            self.encoder = encoder
        elif getattr(cfg.retrieval, "encoder", "clip") == "mock":
            self.encoder = SummaryEncoder(cfg.retrieval.embedding_dim)
        else:
            self.encoder = ClipEncoder(cfg.retrieval.embed_model)
        self._keys: List[str] = []
        self._vectors: List[np.ndarray] = []

    def build(self, samples: List[StanceSample]) -> None:
        self._keys = [s.key() for s in samples]
        self._vectors = [self.encoder.encode_sample(s, self._image_resolver(s)) for s in samples]
        logger.info("Built retrieval index over %d samples.", len(samples))

    def search(self, query: StanceSample, k: int) -> List[tuple[float, int]]:
        """Return (similarity, index) pairs for the top-k nearest neighbours."""
        if not self._vectors:
            return []
        q = self.encoder.encode_sample(query, self._image_resolver(query))
        mat = np.stack(self._vectors)
        sims = mat @ q
        k = min(k, len(sims))
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), int(i)) for i in idx]
