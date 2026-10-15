"""OpenAI-compatible chat and embedding clients plus the protocols the pipeline depends on."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import openai

from envforge.config import ConfigError, EmbedderConfig, RoleConfig


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ChatModel(Protocol):
    model: str

    def chat(
        self, messages: list[dict], *, temperature: float | None = None, max_tokens: int | None = None
    ) -> ChatResult: ...


class EmbedModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _make_client(base_url: str, api_key_env: str, model: str) -> Any:
    key = os.environ.get(api_key_env)
    if key is None:
        raise ConfigError(f"env var {api_key_env} not set for model {model}")
    return openai.OpenAI(base_url=base_url, api_key=key, max_retries=5, timeout=300)


class LLM:
    """Chat client for one configured role; the HTTP client is created on first use."""

    def __init__(self, cfg: RoleConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _make_client(self.cfg.base_url, self.cfg.api_key_env, self.cfg.model)
        return self._client

    def chat(
        self, messages: list[dict], *, temperature: float | None = None, max_tokens: int | None = None
    ) -> ChatResult:
        resp = self._get_client().chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
        )
        usage = resp.usage
        in_tok = (usage.prompt_tokens or 0) if usage is not None else 0
        out_tok = (usage.completion_tokens or 0) if usage is not None else 0
        text = ""
        if resp.choices:
            text = resp.choices[0].message.content or ""
        cost = in_tok * self.cfg.price_in_per_mtok / 1e6 + out_tok * self.cfg.price_out_per_mtok / 1e6
        return ChatResult(text=text, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)


class Embedder:
    """Embedding client; the HTTP client is created on first use."""

    def __init__(self, cfg: EmbedderConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _make_client(self.cfg.base_url, self.cfg.api_key_env, self.cfg.model)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._get_client().embeddings.create(model=self.cfg.model, input=texts)
        items = sorted(resp.data, key=lambda d: d.index)
        return [list(d.embedding) for d in items]
