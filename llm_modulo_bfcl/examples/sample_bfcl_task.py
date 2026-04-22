"""A minimal BFCL-style task used by `main.py` and tests.

Two-function multi-step plan: fetch weather, then email a summary.
"""

SAMPLE_TASK = {
    "id": "demo_weather_email",
    "query": (
        "Look up today's weather in Phoenix and send a brief summary email to "
        "user@example.com with the subject 'Weather Report'."
    ),
    "functions": [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["fahrenheit", "celsius"],
                        "description": "Temperature units",
                    },
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
                    "to": {"type": "string", "format": "email"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    ],
}
