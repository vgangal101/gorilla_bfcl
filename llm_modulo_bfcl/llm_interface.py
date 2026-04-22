"""LLM abstraction. Swap `MockLLM` for a real client (OpenAI/Anthropic/vLLM)."""

from abc import ABC, abstractmethod


class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return raw model output."""


class MockLLM(LLMInterface):
    """Deterministic mock driven by a canned sequence of responses."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            # Repeat final response if we run out. Loop will still terminate via max_iters.
            resp = self.responses[-1] if self.responses else "{}"
        self.call_count += 1
        return resp


class CallableLLM(LLMInterface):
    """Adapter that wraps any `(prompt: str) -> str` callable as an LLMInterface."""

    def __init__(self, fn):
        self._fn = fn

    def generate(self, prompt: str) -> str:
        return self._fn(prompt)
