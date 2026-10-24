"""Shared, ROS-free logic: score one LLM response and integrate its ideal path.

Values the model produced are recorded as they are, even when the validator
rejects them: an out-of-range speed is itself an experimental result.
"""

import math

from llm_velocity_controller.validator import parse_json
from llm_velocity_controller.validator import validate_dict
from llm_velocity_controller.validator import ValidationError

FIELDS = ('linear_x', 'angular_z', 'duration')
EPS = 1e-9
MAX_SAMPLES_PER_ACTION = 200
SAMPLE_DT = 0.05


def _finite_number(value):
    """Mirror validator semantics: int/float but not bool, and finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def extract_twist(obj):
    """dict -> (linear_x, angular_z, duration) floats, or None if not well-formed."""
    if not isinstance(obj, dict) or set(obj) != set(FIELDS):
        return None
    values = tuple(_finite_number(obj[name]) for name in FIELDS)
    return None if any(v is None for v in values) else values


def integrate(actions, start=(0.0, 0.0, 0.0)):
    """Exact unicycle integration of constant-velocity segments.

    actions: iterable of (linear_x, angular_z, duration). Returns a list of
    [t, x, y, theta] samples starting with the start pose. Durations are used as
    given (negative or huge values only change sampling, never the maths).
    """
    x, y, theta = start
    t = 0.0
    points = [[t, x, y, theta]]
    for v, w, duration in actions:
        if duration <= 0.0:
            continue
        steps = int(min(MAX_SAMPLES_PER_ACTION, max(1, math.ceil(duration / SAMPLE_DT))))
        dt = duration / steps
        for _ in range(steps):
            if abs(w) < EPS:
                x += v * math.cos(theta) * dt
                y += v * math.sin(theta) * dt
            else:
                x += v / w * (math.sin(theta + w * dt) - math.sin(theta))
                y -= v / w * (math.cos(theta + w * dt) - math.cos(theta))
            theta += w * dt
            t += dt
            points.append([t, x, y, theta])
    return points


def segment_metrics(v, w, duration):
    distance = v * duration
    rotation = w * duration
    radius = abs(v / w) if abs(w) > EPS and abs(v) > EPS else None
    return {
        'estimated_distance': distance,
        'estimated_rotation_rad': rotation,
        'estimated_rotation_deg': math.degrees(rotation),
        'radius': radius,
    }


def _limit_violations(v, w, duration, limits):
    violations = []
    if abs(v) > limits['linear_x_limit']:
        violations.append('linear_x')
    if abs(w) > limits['angular_z_limit']:
        violations.append('angular_z')
    if duration < 0.0 or duration > limits['duration_max']:
        violations.append('duration')
    return violations


def _empty_result():
    return {
        'json_valid': False, 'validation_passed': False, 'validation_error': '',
        'refused': None, 'limit_violations': [],
        'linear_x': None, 'angular_z': None, 'duration': None,
        'estimated_distance': None, 'estimated_rotation_rad': None,
        'estimated_rotation_deg': None, 'radius': None,
        'n_actions': None, 'actions': None, 'path_length': None,
        'closure_distance': None, 'path': None,
    }


def evaluate_single(content, limits):
    """Score a one-Twist response (raw text from the model)."""
    result = _empty_result()
    try:
        obj = parse_json(content)
    except ValidationError as error:
        result['validation_error'] = str(error)
        return result
    twist = extract_twist(obj)
    if twist is None:
        try:
            validate_dict(obj)          # produces the precise reason
        except ValidationError as error:
            result['validation_error'] = str(error)
        return result
    v, w, duration = twist
    result.update(json_valid=True, linear_x=v, angular_z=w, duration=duration,
                  refused=(v == 0.0 and w == 0.0), n_actions=1,
                  limit_violations=_limit_violations(v, w, duration, limits),
                  **segment_metrics(v, w, duration))
    try:
        validate_dict(obj)
        result['validation_passed'] = True
    except ValidationError as error:
        result['validation_error'] = str(error)
    path = integrate([twist])
    result['path'] = path
    result['path_length'] = abs(v) * max(duration, 0.0)
    result['closure_distance'] = math.hypot(path[-1][1], path[-1][2])
    return result


def evaluate_actions(content, limits):
    """Score an {"actions": [...]} response."""
    result = _empty_result()
    try:
        obj = parse_json(content)
    except ValidationError as error:
        result['validation_error'] = str(error)
        return result
    if set(obj) != {'actions'} or not isinstance(obj['actions'], list):
        result['validation_error'] = 'response must be {"actions": [ ... ]} and nothing else'
        return result
    twists = [extract_twist(item) for item in obj['actions']]
    if any(t is None for t in twists):
        bad = next(i for i, t in enumerate(twists) if t is None)
        try:
            validate_dict(obj['actions'][bad])
            reason = 'not a valid action'
        except ValidationError as error:
            reason = str(error)
        result['validation_error'] = f'action {bad}: {reason}'
        return result

    errors = []
    violations = set()
    for index, (item, twist) in enumerate(zip(obj['actions'], twists)):
        violations.update(_limit_violations(*twist, limits))
        try:
            validate_dict(item)
        except ValidationError as error:
            errors.append(f'action {index}: {error}')

    total_duration = sum(t[2] for t in twists)
    rotation = sum(t[1] * t[2] for t in twists)
    distance = sum(t[0] * t[2] for t in twists)
    path = integrate(twists)
    result.update(
        json_valid=True,
        validation_passed=not errors,
        validation_error='; '.join(errors),
        refused=(len(twists) == 0 or all(t[0] == 0.0 and t[1] == 0.0 for t in twists)),
        limit_violations=sorted(violations),
        n_actions=len(twists),
        actions=[dict(zip(FIELDS, t)) for t in twists],
        duration=total_duration,
        estimated_distance=distance,
        estimated_rotation_rad=rotation,
        estimated_rotation_deg=math.degrees(rotation),
        path_length=sum(abs(t[0]) * max(t[2], 0.0) for t in twists),
        closure_distance=math.hypot(path[-1][1], path[-1][2]),
        path=path,
    )
    if len(twists) == 1:
        v, w, _ = twists[0]
        result.update(linear_x=v, angular_z=w, radius=segment_metrics(*twists[0])['radius'])
    return result


def evaluate(content, mode, limits):
    if mode == 'single':
        return evaluate_single(content, limits)
    if mode == 'actions':
        return evaluate_actions(content, limits)
    raise ValueError(f'unknown mode {mode!r}')


def classify_segments(actions):
    """Split a sequence into straight lines, in-place turns and arcs."""
    straights, turns, arcs = [], [], []
    for a in actions or ():
        v, w, d = a['linear_x'], a['angular_z'], a['duration']
        if d <= 0.0 or (abs(v) < EPS and abs(w) < EPS):
            continue
        if abs(w) < EPS:
            straights.append(abs(v) * d)
        elif abs(v) < EPS:
            turns.append(math.degrees(w * d))
        else:
            arcs.append({'radius': abs(v / w), 'angle_deg': math.degrees(w * d)})
    return straights, turns, arcs


def turn_direction_changes(actions):
    """How many times the turning direction flips along the sequence."""
    signs = [math.copysign(1.0, a['angular_z']) for a in actions or ()
             if abs(a['angular_z']) > EPS and a['duration'] > 0.0]
    return sum(1 for s0, s1 in zip(signs, signs[1:]) if s0 != s1)
