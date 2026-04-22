"""Framework configuration.

Hard critics gate acceptance: if any hard critic fails, the plan is rejected.
Soft critics produce feedback but do not block acceptance.
"""

MAX_ITERATIONS: int = 10

HARD_CRITICS: set[str] = {
    "ParserCritic",
    "FunctionValidityCritic",
    "ArgumentSchemaCritic",
    "ArgumentValueCritic",
    "DependencyCritic",
    "ExecutionCritic",
}

SOFT_CRITICS: set[str] = {
    "RedundancyCritic",
}


def is_hard_critic(name: str) -> bool:
    return name in HARD_CRITICS
