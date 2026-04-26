"""RuntimeCritic: actually invoke the tool and pass on the output through metadata.

The critic both validates ("does it run without raising?") and produces the
real tool output. The runner reads `metadata["output"]` from the result and
uses it directly — no double-execution.

If the tool is side-effectful (e.g. sends an email), this critic invokes
the side effect even on iterations that ultimately get rejected by other
critics; in practice the demo tools are pure, and for production a caller
would wrap side-effectful tools in a dry-run/commit pair before exposing
them.
"""

from .base import MultiTurnCritic
from ..schemas import MultiTurnCriticResult, Proposal


class RuntimeCritic(MultiTurnCritic):
    name = "RuntimeCritic"

    def evaluate(self, proposal: Proposal, context: dict) -> MultiTurnCriticResult:
        if proposal.type != "function_call":
            return self._pass()

        registry = context.get("tool_registry") or {}
        fn = registry.get(proposal.function_name)
        if fn is None:
            return self._fail(
                f"no implementation registered for '{proposal.function_name}'"
            )

        args = proposal.arguments or {}
        try:
            output = fn(**args)
        except TypeError as e:
            # signature mismatch the schema critic missed (e.g. wrong kw name)
            return self._fail(
                f"signature error calling '{proposal.function_name}': {e}"
            )
        except Exception as e:  # noqa: BLE001 — surface any tool failure
            return self._fail(
                f"runtime error in '{proposal.function_name}': "
                f"{type(e).__name__}: {e}"
            )

        return self._pass(metadata={"output": output})
