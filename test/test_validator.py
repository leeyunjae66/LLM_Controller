import json
import math

import pytest

from llm_velocity_controller.validator import validate
from llm_velocity_controller.validator import validate_dict
from llm_velocity_controller.validator import ValidationError
from llm_velocity_controller.validator import VelocityCommand


def _json(**fields):
    return json.dumps(fields)


@pytest.mark.parametrize('text, expected', [
    (_json(linear_x=0.2, angular_z=0.0, duration=1.0), (0.2, 0.0, 1.0)),
    (_json(linear_x=-0.1, angular_z=0, duration=1), (-0.1, 0.0, 1.0)),
    (_json(linear_x=0, angular_z=0.3, duration=1), (0.0, 0.3, 1.0)),
    (_json(linear_x=0, angular_z=-0.3, duration=2), (0.0, -0.3, 2.0)),
    (_json(linear_x=0, angular_z=0, duration=0), (0.0, 0.0, 0.0)),
    # inclusive boundaries
    (_json(linear_x=-0.2, angular_z=0.5, duration=2.0), (-0.2, 0.5, 2.0)),
    (_json(linear_x=0.2, angular_z=-0.5, duration=0.0), (0.2, -0.5, 0.0)),
])
def test_accepts_valid_commands(text, expected):
    command = validate(text)
    assert (command.linear_x, command.angular_z, command.duration) == expected
    assert all(isinstance(v, float) for v in expected)


def test_stop_command_is_recognised():
    assert validate(_json(linear_x=0, angular_z=0, duration=0)).is_stop
    assert VelocityCommand.stop().is_stop
    assert not validate(_json(linear_x=0.1, angular_z=0, duration=1)).is_stop


def test_negative_zero_is_normalised():
    command = validate('{"linear_x": -0.0, "angular_z": -0.0, "duration": 1}')
    assert math.copysign(1.0, command.linear_x) == 1.0
    assert math.copysign(1.0, command.angular_z) == 1.0


@pytest.mark.parametrize('text', [
    '',
    'not json',
    '{"linear_x": 0.1, "angular_z": 0, "duration": 1',      # truncated
    '{"linear_x": 0.1, "angular_z": 0, "duration": 1} xx',  # trailing junk
    "{'linear_x': 0.1, 'angular_z': 0, 'duration': 1}",     # single quotes
    '```json\n{"linear_x": 0.1, "angular_z": 0, "duration": 1}\n```',
])
def test_rejects_unparseable_json(text):
    with pytest.raises(ValidationError, match='invalid JSON'):
        validate(text)


@pytest.mark.parametrize('text', ['[0.1, 0, 1]', '0.1', '"stop"', 'null', 'true'])
def test_rejects_non_object(text):
    with pytest.raises(ValidationError, match='object'):
        validate(text)


def test_rejects_non_string_input():
    with pytest.raises(ValidationError, match='string'):
        validate(b'{"linear_x": 0, "angular_z": 0, "duration": 0}')


@pytest.mark.parametrize('missing', ['linear_x', 'angular_z', 'duration'])
def test_rejects_missing_field(missing):
    fields = {'linear_x': 0.1, 'angular_z': 0.0, 'duration': 1.0}
    del fields[missing]
    with pytest.raises(ValidationError, match=f'missing field.*{missing}'):
        validate(json.dumps(fields))


def test_rejects_extra_field():
    with pytest.raises(ValidationError, match='unexpected field.*linear_y'):
        validate(_json(linear_x=0.1, angular_z=0, duration=1, linear_y=0.1))


def test_rejects_duplicate_field():
    with pytest.raises(ValidationError, match='duplicate field: linear_x'):
        validate('{"linear_x": 0.1, "linear_x": 5, "angular_z": 0, "duration": 1}')


@pytest.mark.parametrize('value', ['true', 'false', '"0.1"', 'null', '[0.1]', '{}'])
def test_rejects_non_numeric_types(value):
    text = f'{{"linear_x": {value}, "angular_z": 0, "duration": 1}}'
    with pytest.raises(ValidationError, match='linear_x must be a number'):
        validate(text)


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity'])
def test_rejects_non_finite_constants(constant):
    text = f'{{"linear_x": 0.1, "angular_z": {constant}, "duration": 1}}'
    with pytest.raises(ValidationError, match='non-finite'):
        validate(text)


def test_rejects_non_finite_values_in_dict():
    for bad in (float('nan'), float('inf'), float('-inf')):
        with pytest.raises(ValidationError, match='finite'):
            validate_dict({'linear_x': bad, 'angular_z': 0.0, 'duration': 1.0})


def test_rejects_overflowing_number():
    # 1e400 parses to inf in Python; it must not slip through.
    with pytest.raises(ValidationError):
        validate('{"linear_x": 1e400, "angular_z": 0, "duration": 1}')


@pytest.mark.parametrize('fields, message', [
    ({'linear_x': 10.0, 'angular_z': 0, 'duration': 1}, 'linear_x'),    # 앞으로 10m/s
    ({'linear_x': 0.2000001, 'angular_z': 0, 'duration': 1}, 'linear_x'),
    ({'linear_x': -0.3, 'angular_z': 0, 'duration': 1}, 'linear_x'),
    ({'linear_x': 0, 'angular_z': 0.51, 'duration': 1}, 'angular_z'),
    ({'linear_x': 0, 'angular_z': -1.0, 'duration': 1}, 'angular_z'),
    ({'linear_x': 0.1, 'angular_z': 0, 'duration': 2.01}, 'duration'),
    ({'linear_x': 0.1, 'angular_z': 0, 'duration': 60}, 'duration'),
])
def test_rejects_out_of_range_without_clamping(fields, message):
    with pytest.raises(ValidationError, match=message):
        validate(json.dumps(fields))


def test_rejects_negative_duration():
    with pytest.raises(ValidationError, match='must not be negative'):
        validate(_json(linear_x=0.1, angular_z=0, duration=-1))
