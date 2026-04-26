"""Collapse a list of critic results into a bullet-list of failures."""

from .schemas import MultiTurnCriticResult


def aggregate(results: list[MultiTurnCriticResult]) -> str:
    failures = [r for r in results if not r.passed]
    if not failures:
        return ""
    return "\n".join(f"- [{r.name}] {r.feedback}" for r in failures)
