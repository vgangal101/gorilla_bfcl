"""Multi-turn LLM-Modulo demo.

Two scenarios driven by a deterministic MockLLM:

  Scenario A: user supplies all info up front; agent calls two tools and
              terminates with a final_answer.
  Scenario B: user is vague; agent emits a clarification proposal; user
              responds in a second message; agent then proceeds to call
              tools and finalize.

Run from inside `llm_modulo_bfcl/`:

    python multi_turn_main.py
"""

import json

from llm_interface import MockLLM
from multi_turn import (
    MultiTurnMetaController,
    ProposalParser,
    run_multi_turn,
)
from multi_turn.critics import (
    ContextGroundingCritic,
    MissingInformationCritic,
    RuntimeCritic,
    SchemaCritic,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_weather(city: str, units: str = "fahrenheit") -> dict:
    return {"city": city, "temp": 102, "conditions": "sunny", "units": units}


def send_email(to: str, subject: str, body: str) -> dict:
    return {"delivered": True, "to": to, "subject": subject}


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "send_email": send_email,
}

TOOL_SPECS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["fahrenheit", "celsius"]},
            },
            "required": ["city"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


def _critics():
    return [
        SchemaCritic(),
        ContextGroundingCritic(),
        MissingInformationCritic(),
        RuntimeCritic(),
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _drive(llm: MockLLM, user_messages: list[str]) -> None:
    for event in run_multi_turn(
        user_messages=iter(user_messages),
        llm=llm,
        parser=ProposalParser(),
        critics=_critics(),
        meta=MultiTurnMetaController(),
        tool_specs=TOOL_SPECS,
        tool_registry=TOOL_REGISTRY,
        max_modulo_iters=5,
        max_steps_per_turn=10,
    ):
        ev = event["event"]
        if ev == "user_message":
            print(f"\n>>> USER : {event['message']}")
        elif ev == "tool_call":
            print(
                f"    TOOL : {event['function']}({event['arguments']}) "
                f"-> {event['output']}"
            )
        elif ev == "clarification":
            print(f"<<< AGENT (asks)  : {event['message']}")
        elif ev == "final_answer":
            print(f"<<< AGENT (final) : {event['message']}")
        elif ev == "error":
            print(f"!!! ERROR : {event['message']}")
        elif ev == "end_of_user_messages":
            print("    (end of conversation, no final_answer reached)")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_a() -> None:
    print("=" * 70)
    print("Scenario A: full info up front -> tool, tool, final_answer")
    print("=" * 70)

    llm = MockLLM(responses=[
        json.dumps({
            "type": "function_call",
            "function_name": "get_weather",
            "arguments": {"city": "Phoenix"},
        }),
        json.dumps({
            "type": "function_call",
            "function_name": "send_email",
            "arguments": {
                "to": "user@example.com",
                "subject": "Weather Report",
                "body": "Phoenix today: 102F, sunny.",
            },
        }),
        json.dumps({
            "type": "final_answer",
            "message": "Sent the Phoenix weather summary to user@example.com.",
        }),
    ])

    _drive(llm, [
        "What's the weather in Phoenix and email a summary to user@example.com?",
    ])


def scenario_b() -> None:
    print("\n" + "=" * 70)
    print("Scenario B: vague request -> clarification -> resolution")
    print("=" * 70)

    llm = MockLLM(responses=[
        # 1st user message ("Send a weather email."): agent must ask back.
        json.dumps({
            "type": "clarification",
            "message": "Which city's weather should I include, "
                       "and what email address should I send it to?",
        }),
        # 2nd user message ("Phoenix, to user@example.com"): proceed.
        json.dumps({
            "type": "function_call",
            "function_name": "get_weather",
            "arguments": {"city": "Phoenix"},
        }),
        json.dumps({
            "type": "function_call",
            "function_name": "send_email",
            "arguments": {
                "to": "user@example.com",
                "subject": "Weather Update",
                "body": "Phoenix: 102F, sunny.",
            },
        }),
        json.dumps({
            "type": "final_answer",
            "message": "Done — emailed the Phoenix weather to user@example.com.",
        }),
    ])

    _drive(llm, [
        "Send a weather email.",
        "Phoenix, to user@example.com",
    ])


def main() -> None:
    scenario_a()
    scenario_b()


if __name__ == "__main__":
    main()
