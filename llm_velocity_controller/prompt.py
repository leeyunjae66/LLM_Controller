"""Prompt construction and local STOP detection."""

import re

from llm_velocity_controller.validator import ANGULAR_Z_LIMIT
from llm_velocity_controller.validator import DURATION_MAX
from llm_velocity_controller.validator import LINEAR_X_LIMIT

DEFAULT_LINEAR_SPEED = 0.1    # m/s
DEFAULT_ANGULAR_SPEED = 0.3   # rad/s
DEFAULT_DURATION = 1.0        # s

# JSON Schema passed to Ollama's `format` for structured output. Ranges are
# deliberately not encoded here: out-of-range requests must reach the
# validator verbatim so they are rejected instead of silently clamped.
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

SYSTEM_PROMPT = f"""You convert a robot movement command (usually Korean) into JSON.
Respond with exactly one JSON object and nothing else:
{{"linear_x": <number>, "angular_z": <number>, "duration": <number>}}

Fields:
- linear_x: forward speed in m/s. Forward/앞으로 is positive, backward/뒤로 is negative.
- angular_z: yaw rate in rad/s. Left/왼쪽 (counter-clockwise) is positive,
  right/오른쪽 (clockwise) is negative.
- duration: seconds to keep moving.

Allowed range: linear_x {-LINEAR_X_LIMIT}..{LINEAR_X_LIMIT} m/s,
angular_z {-ANGULAR_Z_LIMIT}..{ANGULAR_Z_LIMIT} rad/s, duration 0..{DURATION_MAX} s.

Rules:
1. If a speed or duration is stated, copy that number exactly, even if it is
   outside the allowed range. Never clamp, round or change a stated value.
2. If no speed is stated: use {DEFAULT_LINEAR_SPEED} m/s for forward/backward and
   {DEFAULT_ANGULAR_SPEED} rad/s for turning.
3. If no duration is stated, use {DEFAULT_DURATION}.
4. Moving straight means angular_z = 0. Turning in place means linear_x = 0.
5. Stop commands (정지, 멈춰, stop) and any unclear, unsafe or unsupported command
   (sideways, up/down, questions, unknown words) must return
   {{"linear_x": 0, "angular_z": 0, "duration": 0}}.

Examples:
앞으로 1초 동안 0.2m/s로 이동해 -> {{"linear_x": 0.2, "angular_z": 0, "duration": 1}}
뒤로 1초 동안 0.1m/s로 이동해 -> {{"linear_x": -0.1, "angular_z": 0, "duration": 1}}
왼쪽으로 1초 동안 0.3rad/s로 회전해 -> {{"linear_x": 0, "angular_z": 0.3, "duration": 1}}
오른쪽으로 2초 동안 회전해 -> {{"linear_x": 0, "angular_z": -{DEFAULT_ANGULAR_SPEED}, "duration": 2}}
앞으로 이동해 -> {{"linear_x": {DEFAULT_LINEAR_SPEED}, "angular_z": 0, "duration": {DEFAULT_DURATION}}}
앞으로 5m/s로 가 -> {{"linear_x": 5, "angular_z": 0, "duration": {DEFAULT_DURATION}}}
정지해 -> {{"linear_x": 0, "angular_z": 0, "duration": 0}}
노래 불러줘 -> {{"linear_x": 0, "angular_z": 0, "duration": 0}}
"""

# Matched locally so STOP never waits for LLM inference. Deliberately broad:
# a false positive only stops the robot.
_STOP_PATTERN = re.compile(
    r'정지|멈춰|멈춤|멈추|스톱|그만|비상|서라|\bstop\b|\bhalt\b|\bestop\b|e-stop',
    re.IGNORECASE)


def is_stop_command(text):
    """True if the command must stop the robot immediately without the LLM."""
    if text is None or not text.strip():
        return True
    return _STOP_PATTERN.search(text) is not None


def build_messages(user_command):
    """Return the Ollama /api/chat `messages` list for one user command."""
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_command.strip()},
    ]
