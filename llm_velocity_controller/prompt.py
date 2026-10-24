
"""Prompt construction and local STOP detection."""

import re

from llm_velocity_controller.validator import ANGULAR_Z_LIMIT
from llm_velocity_controller.validator import DURATION_MAX
from llm_velocity_controller.validator import LINEAR_X_LIMIT

# Retained for compatibility with MockLLMClient.
# The real LLM prompt does not force these defaults.
DEFAULT_LINEAR_SPEED = 0.1
DEFAULT_ANGULAR_SPEED = 0.3
DEFAULT_DURATION = 5.0

# Structured JSON output
RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'linear_x': {'type': 'number'},
        'angular_z': {'type': 'number'},
        'duration': {'type': 'number'},
    },
    'required': ['linear_x', 'angular_z', 'duration'],
    'additionalProperties': False,
}

SYSTEM_PROMPT = f"""
You are an intelligent natural-language motion interpreter
for a mobile robot.

Your task is to understand the user's movement intention
and autonomously generate appropriate numerical motion commands.

Output exactly one JSON object:
{{"linear_x": <number>, "angular_z": <number>, "duration": <number>}}

Motion parameters:
- linear_x: forward/backward velocity in m/s.
  Positive is forward, negative is backward.
- angular_z: rotational velocity in rad/s.
  Positive is counter-clockwise, negative is clockwise.
- duration: movement duration in seconds.

Interpret the user's instruction using your own
language understanding and reasoning.

Autonomously determine suitable velocity and duration
based on the user's intended movement.

Do not use fixed word-to-number mappings.
Do not assign identical motion parameters to commands
that imply meaningfully different movement intensities.

Distinguish between intended movement speed,
movement magnitude, direction, and duration.

Choose physically meaningful numerical values.

For values that are not explicitly specified,
choose them autonomously within these limits:
- linear_x: {-LINEAR_X_LIMIT} to {LINEAR_X_LIMIT} m/s
- angular_z: {-ANGULAR_Z_LIMIT} to {ANGULAR_Z_LIMIT} rad/s
- duration: 0 to {DURATION_MAX} seconds

If the user explicitly specifies a numerical speed
or duration, preserve the requested value exactly,
even when it exceeds these limits.
Do not silently clamp an explicit value.
A separate validator will reject invalid commands.

If the command is unclear, unsupported, or unsafe,
output zero velocities and zero duration.

If the user requests an indefinite movement,
do not generate an indefinite command.
Use a bounded duration or return a stop command
when safe interpretation is not possible.

Return only the JSON object.
"""

# Local STOP detection without LLM inference
_STOP_PATTERN = re.compile(
    r'정지|멈춰|멈춤|멈추|스톱|그만|비상|서라|\bstop\b|\bhalt\b|\bestop\b|e-stop',
    re.IGNORECASE
)


def is_stop_command(text):
    """Detect stop commands locally."""
    if text is None or not text.strip():
        return True
    return _STOP_PATTERN.search(text) is not None


def build_messages(user_command):
    """Construct messages for Ollama."""
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_command.strip()},
    ]
