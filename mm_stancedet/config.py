"""Configuration loading for MM-StanceDet.

Reads config/config.yaml and exposes a typed Config object. All paths given in
the YAML file are interpreted relative to the project root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com/chat/completions"
    model: str = "deepseek-v4-flash-vision-exp"
    api_key: str = ""
    api_key_file: str = "config/api_key.txt"
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_sec: int = 120
    max_retries: int = 3


@dataclass
class DataConfig:
    root: str = "data/stance"
    annotations: str = "annotations.jsonl"
    images_dir: str = "images"
    retrieval_split: str = "valid"
    eval_split: str = "test"


@dataclass
class RetrievalConfig:
    top_k: int = 3
    embed_model: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    encoder: str = "clip"


@dataclass
class DebateConfig:
    rounds: int = 3


@dataclass
class OutputConfig:
    dir: str = "outputs"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def resolve(self, rel_path: str) -> Path:
        """Resolve a config-relative path against the project root."""
        return PROJECT_ROOT / rel_path

    def api_key(self) -> Optional[str]:
        """Return the API key from config.yaml, falling back to a local file."""
        if self.llm.api_key:
            return self.llm.api_key.strip()
        key_file = self.resolve(self.llm.api_key_file)
        if key_file.exists():
            return key_file.read_text(encoding="utf-8").strip()
        return None


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Read the YAML config and build a typed Config object."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    def sub(key, cls):
        return cls(**raw.get(key, {}))

    cfg = Config(
        llm=sub("llm", LLMConfig),
        data=sub("data", DataConfig),
        retrieval=sub("retrieval", RetrievalConfig),
        debate=sub("debate", DebateConfig),
        output=sub("output", OutputConfig),
    )
    return cfg
