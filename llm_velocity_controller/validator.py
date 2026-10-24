"""Validation of LLM velocity JSON. Pure Python; no ROS dependency.

Values outside the allowed range are rejected, never clamped.
"""

import json
import math
from dataclasses import dataclass

LINEAR_X_LIMIT = 1.0     # m/s, symmetric
ANGULAR_Z_LIMIT = 0.5     # rad/s, symmetric
DURATION_MAX = 10.0        # s
REQUIRED_FIELDS = ('linear_x', 'angular_z', 'duration')


class ValidationError(ValueError):
    """Raised when an LLM response must not be executed."""


@dataclass(frozen=True)
class VelocityCommand:
    linear_x: float
    angular_z: float
    duration: float

    @classmethod
    def stop(cls):
        return cls(0.0, 0.0, 0.0)

    @property
    def is_stop(self):
        return self.linear_x == 0.0 and self.angular_z == 0.0


def _reject_constant(name):
    # json.loads accepts NaN / Infinity / -Infinity by default.
    raise ValidationError(f'non-finite constant {name} is not allowed')


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f'duplicate field: {key}')
        result[key] = value
    return result


def parse_json(text):
    """Parse raw LLM output into a dict, rejecting anything ambiguous."""
    if not isinstance(text, str):
        raise ValidationError(f'response must be a string, got {type(text).__name__}')
    try:
        data = json.loads(text, parse_constant=_reject_constant,
                          object_pairs_hook=_reject_duplicates)
    except ValidationError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValidationError(f'invalid JSON: {error}') from None
    if not isinstance(data, dict):
        raise ValidationError('top-level JSON value must be an object')
    return data


def _number(data, field):
    value = data[field]
    # bool is a subclass of int; it must not pass as a number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f'{field} must be a number, got {type(value).__name__}')
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f'{field} must be finite')
    return value


def validate_dict(data):
    """Validate an already-parsed object and return a VelocityCommand."""
    if not isinstance(data, dict):
        raise ValidationError('command must be a JSON object')
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValidationError(f'missing field(s): {", ".join(missing)}')
    extra = sorted(set(data) - set(REQUIRED_FIELDS))
    if extra:
        raise ValidationError(f'unexpected field(s): {", ".join(extra)}')

    linear_x = _number(data, 'linear_x')
    angular_z = _number(data, 'angular_z')
    duration = _number(data, 'duration')

    if abs(linear_x) > LINEAR_X_LIMIT:
        raise ValidationError(
            f'linear_x {linear_x} outside [-{LINEAR_X_LIMIT}, {LINEAR_X_LIMIT}] m/s')
    if abs(angular_z) > ANGULAR_Z_LIMIT:
        raise ValidationError(
            f'angular_z {angular_z} outside [-{ANGULAR_Z_LIMIT}, {ANGULAR_Z_LIMIT}] rad/s')
    if duration < 0.0:
        raise ValidationError(f'duration {duration} must not be negative')
    if duration > DURATION_MAX:
        raise ValidationError(f'duration {duration} exceeds {DURATION_MAX} s')

    # Normalise -0.0 so downstream equality checks stay simple.
    return VelocityCommand(linear_x + 0.0, angular_z + 0.0, duration + 0.0)


def validate(text):
    """Parse and validate raw LLM output. Raises ValidationError on any problem."""
    return validate_dict(parse_json(text))
