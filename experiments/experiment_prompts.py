"""System prompts and JSON Schemas for the evaluation experiments.

Kept separate from the robot-control prompt in llm_velocity_controller/prompt.py
so experiments never change what the ROS node sends to the LLM.

Design rules (see README):
- explain what each output field physically means and its sign convention;
- give NO example values, default values or word-to-number mappings, so the
  model has to choose every number itself;
- never mention the test commands, their expected order or reference ranges;
- tell the model to keep explicit numbers verbatim and not to clamp;
- give an explicit way to say "this cannot be represented" instead of
  forcing a fake motion.

A, B, C and D1 currently share one base text on purpose: then differences
between experiments come from the commands, not from prompt wording. Each
experiment still has its own entry so it can be changed independently.
"""

from llm_velocity_controller.validator import ANGULAR_Z_LIMIT
from llm_velocity_controller.validator import DURATION_MAX
from llm_velocity_controller.validator import LINEAR_X_LIMIT

PROMPT_VARIANTS = ('with_limits', 'no_limits')
MAX_ACTIONS = 30

_TWIST_PROPERTIES = {
    'linear_x': {'type': 'number'},
    'angular_z': {'type': 'number'},
    'duration': {'type': 'number'},
}

# Ranges are intentionally not part of the schemas: an out-of-range choice must
# reach the validator unchanged so it is measured, not hidden by the grammar.
SINGLE_SCHEMA = {
    'type': 'object',
    'properties': dict(_TWIST_PROPERTIES),
    'required': ['linear_x', 'angular_z', 'duration'],
    'additionalProperties': False,
}

ACTIONS_SCHEMA = {
    'type': 'object',
    'properties': {
        'actions': {
            'type': 'array',
            'maxItems': MAX_ACTIONS,
            'items': {
                'type': 'object',
                'properties': dict(_TWIST_PROPERTIES),
                'required': ['linear_x', 'angular_z', 'duration'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['actions'],
    'additionalProperties': False,
}

_FIELDS = """Output fields:
- linear_x: forward velocity in m/s. Positive moves forward, negative moves backward.
- angular_z: yaw rate in rad/s. Positive turns left (counter-clockwise, seen from
  above), negative turns right (clockwise).
- duration: time in seconds during which linear_x and angular_z are held constant.
  The robot stops afterwards.

Physical meaning: while a command runs, the robot travels linear_x * duration metres
and turns angular_z * duration radians. With both linear_x and angular_z non-zero it
drives along a circular arc of radius |linear_x / angular_z| metres."""


def _limits_text(variant):
    if variant == 'no_limits':
        return ''
    return f"""
Robot capability (choose values the robot can actually perform):
- |linear_x| <= {LINEAR_X_LIMIT} m/s
- |angular_z| <= {ANGULAR_Z_LIMIT} rad/s
- 0 <= duration <= {DURATION_MAX} s
"""


_RULES = """Rules:
1. Decide every number yourself from the meaning of the command. No default values
   are given; interpret speed, distance, angle and time words with your own
   understanding of the physical world.
2. If the command states a number with a unit, keep that number exactly, even if it
   is outside the robot capability. Never round, clamp or rescale it; a separate
   validator decides whether it can be executed.
3. Do not pretend: if the command cannot be expressed in this format, or it is
   unclear or unsafe, say so with the stop form described below instead of
   inventing a motion that does something else."""


def _single_prompt(variant):
    return f"""You translate one natural-language command (usually Korean) for a
differential-drive mobile robot on flat ground into a single constant velocity command.

Return exactly one JSON object and nothing else:
{{"linear_x": <number>, "angular_z": <number>, "duration": <number>}}

{_FIELDS}
{_limits_text(variant)}
{_RULES}

Stop form (cannot be represented as ONE constant command, unclear or unsafe):
{{"linear_x": 0, "angular_z": 0, "duration": 0}}
"""


def _actions_prompt(variant):
    return f"""You translate one natural-language command (usually Korean) for a
differential-drive mobile robot on flat ground into a sequence of motion segments.

Return exactly one JSON object and nothing else:
{{"actions": [{{"linear_x": <number>, "angular_z": <number>, "duration": <number>}}, ...]}}

The segments run one after another, in order. Each segment holds its own constant
linear_x and angular_z for its duration. The robot starts at the origin facing the
+x direction. Use as many segments as the motion needs (at most {MAX_ACTIONS}).

{_FIELDS}
{_limits_text(variant)}
{_RULES}

Stop form (cannot be represented as such a sequence, unclear or unsafe):
{{"actions": []}}
"""


def build_prompts(variant='with_limits'):
    """Return {prompt_key: (system_prompt, schema)} for one prompt variant."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f'prompt variant must be one of {PROMPT_VARIANTS}, got {variant!r}')
    single = _single_prompt(variant)
    return {
        'A': (single, SINGLE_SCHEMA),
        'B': (single, SINGLE_SCHEMA),
        'C': (single, SINGLE_SCHEMA),
        'D1': (single, SINGLE_SCHEMA),
        'D2': (_actions_prompt(variant), ACTIONS_SCHEMA),
    }


def current_limits():
    return {
        'linear_x_limit': LINEAR_X_LIMIT,
        'angular_z_limit': ANGULAR_Z_LIMIT,
        'duration_max': DURATION_MAX,
    }
