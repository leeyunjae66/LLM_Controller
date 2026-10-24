"""Offline tests for experiments/ (no Ollama, no ROS)."""

import csv
import json
import math
from pathlib import Path
import re
import sys

import pytest

EXPERIMENTS = Path(__file__).resolve().parent.parent / 'experiments'
sys.path.insert(0, str(EXPERIMENTS))

import analyze_results  # noqa: E402
from experiment_cases import GROUPS  # noqa: E402
from experiment_cases import groups_for  # noqa: E402
from experiment_clients import preflight  # noqa: E402
from experiment_core import classify_segments  # noqa: E402
from experiment_core import evaluate_actions  # noqa: E402
from experiment_core import evaluate_single  # noqa: E402
from experiment_core import integrate  # noqa: E402
from experiment_core import turn_direction_changes  # noqa: E402
from experiment_prompts import build_prompts  # noqa: E402
from experiment_prompts import current_limits  # noqa: E402
from experiment_prompts import MAX_ACTIONS  # noqa: E402
import run_llm_experiment  # noqa: E402

from llm_velocity_controller.llm_client import LLMConnectionError  # noqa: E402
from llm_velocity_controller.llm_client import LLMTimeoutError  # noqa: E402

# Derive test values from validator.py's current limits so the tests hold
# whatever limits the working copy uses.
LIMITS = current_limits()
V = LIMITS['linear_x_limit'] / 2
W = LIMITS['angular_z_limit'] / 2
T = LIMITS['duration_max'] / 2


def _twist(v, w, d):
    return json.dumps({'linear_x': v, 'angular_z': w, 'duration': d})


# ------------------------------------------------------------ cases & prompts

def test_every_group_has_ten_unique_cases_and_valid_relations():
    for group in GROUPS:
        ids = [c.case_id for c in group.cases]
        assert len(ids) == 10 and len(set(ids)) == 10, group.group_id
        for rel in group.relations:
            assert rel.a in ids and rel.b in ids and rel.op in ('<', '<=')
        for case in group.cases:
            if case.reference_range:
                low, high = case.reference_range
                assert 0 < low < high


def test_experiment_selection():
    assert [g.group_id for g in groups_for('A')] == ['A1', 'A2']
    assert [g.group_id for g in groups_for('D')] == ['D1', 'D2']
    assert len(groups_for('all')) == 6
    with pytest.raises(ValueError):
        groups_for('E')


@pytest.mark.parametrize('variant', ['with_limits', 'no_limits'])
def test_prompts_do_not_leak_test_data_or_answer_mappings(variant):
    prompts = build_prompts(variant)
    adverbs = ('천천히', '빠르게', '살살', '느긋', '조심', '잠깐', '잠시', '한동안', '오래',
               '뼘', '발짝', '걸음', '손바닥', '책상', '유턴', '사각형', '지그재그', '포물선')
    for key, (text, _) in prompts.items():
        for group in GROUPS:
            for case in group.cases:
                assert case.command not in text, (key, case.case_id)
        for word in adverbs:
            assert word not in text, (key, word)
        for word in ('e.g.', 'for example', 'example:'):
            assert word not in text.lower(), (key, word)
        # The only numbers allowed are the robot limits, the stop form's zeros
        # and the action cap: any other number would be a value hint.
        allowed = {float(v) for v in current_limits().values()} | {0.0, float(MAX_ACTIONS)}
        body = re.sub(r'(?m)^\s*\d+\.\s', '', text)         # drop list numbering "1. "
        numbers = {float(n) for n in re.findall(r'(?<![\w.])\d+(?:\.\d+)?', body)}
        assert numbers <= allowed, (key, sorted(numbers - allowed))


def test_prompt_variants_differ_only_in_limits():
    with_limits = build_prompts('with_limits')['A'][0]
    no_limits = build_prompts('no_limits')['A'][0]
    limits = current_limits()
    assert f'{limits["linear_x_limit"]} m/s' in with_limits
    assert 'Robot capability' not in no_limits
    with pytest.raises(ValueError):
        build_prompts('other')


def test_schemas_do_not_encode_ranges():
    for _, schema in build_prompts().values():
        assert 'minimum' not in json.dumps(schema) and 'maximum' not in json.dumps(schema)


# --------------------------------------------------------------- evaluation

def test_single_valid_response():
    r = evaluate_single(_twist(V, -W, T), LIMITS)
    assert r['json_valid'] and r['validation_passed'] and not r['refused']
    assert r['estimated_distance'] == pytest.approx(V * T)
    assert r['estimated_rotation_rad'] == pytest.approx(-W * T)
    assert r['estimated_rotation_deg'] == pytest.approx(math.degrees(-W * T))
    assert r['radius'] == pytest.approx(V / W)


def test_single_out_of_range_keeps_values_but_fails_validation():
    fast, long = 4 * V, 4 * T
    r = evaluate_single(_twist(fast, 0, long), LIMITS)
    assert r['json_valid'] and not r['validation_passed']
    assert r['linear_x'] == fast and r['duration'] == long
    assert set(r['limit_violations']) == {'duration', 'linear_x'}


@pytest.mark.parametrize('text', ['', 'nope', '{"linear_x": 0.1}', '[1, 2, 3]',
                                  '{"linear_x": true, "angular_z": 0, "duration": 1}',
                                  '{"linear_x": NaN, "angular_z": 0, "duration": 1}'])
def test_single_malformed(text):
    r = evaluate_single(text, LIMITS)
    assert not r['json_valid'] and not r['validation_passed'] and r['validation_error']


def test_zero_command_is_a_refusal():
    r = evaluate_single('{"linear_x": 0, "angular_z": 0, "duration": 0}', LIMITS)
    assert r['refused'] and r['validation_passed'] and r['radius'] is None


def test_actions_square():
    side = {'linear_x': 0.25, 'angular_z': 0, 'duration': 4}
    turn = {'linear_x': 0, 'angular_z': 0.5, 'duration': math.pi}
    r = evaluate_actions(json.dumps({'actions': [side, turn] * 4}), LIMITS)
    # geometry only: whether these durations pass depends on the current limits
    assert r['json_valid'] and r['n_actions'] == 8
    assert r['path_length'] == pytest.approx(4.0)
    assert r['closure_distance'] == pytest.approx(0.0, abs=1e-6)
    assert r['estimated_rotation_deg'] == pytest.approx(360.0)
    straights, turns, arcs = classify_segments(r['actions'])
    assert straights == pytest.approx([1.0] * 4) and turns == pytest.approx([90.0] * 4)
    assert arcs == []


def test_actions_empty_is_refusal_and_bad_item_is_invalid():
    empty = evaluate_actions('{"actions": []}', LIMITS)
    assert empty['json_valid'] and empty['refused'] and empty['n_actions'] == 0
    bad = evaluate_actions('{"actions": [{"linear_x": 0.1}]}', LIMITS)
    assert not bad['json_valid'] and 'action 0' in bad['validation_error']
    extra = evaluate_actions('{"actions": [], "note": "x"}', LIMITS)
    assert not extra['json_valid']


def test_actions_out_of_range_segment_fails_validation():
    r = evaluate_actions(json.dumps({'actions': [
        {'linear_x': V, 'angular_z': 0, 'duration': T},
        {'linear_x': 0, 'angular_z': 4 * W, 'duration': T}]}), LIMITS)
    assert r['json_valid'] and not r['validation_passed']
    assert 'action 1' in r['validation_error'] and r['limit_violations'] == ['angular_z']


def test_integrate_straight_turn_and_arc():
    end = integrate([(0.5, 0.0, 2.0)])[-1]
    assert end[1:3] == pytest.approx([1.0, 0.0])
    spin = integrate([(0.0, 0.5, math.pi)])[-1]
    assert spin[1:4] == pytest.approx([0.0, 0.0, math.pi / 2])
    # quarter circle to the left with radius 1
    arc = integrate([(0.5, 0.5, math.pi)])[-1]
    assert arc[1:4] == pytest.approx([1.0, 1.0, math.pi / 2], abs=1e-9)


def test_integrate_caps_samples_for_huge_durations():
    assert len(integrate([(0.1, 0.0, 1e6)])) <= 201


def test_turn_direction_changes():
    zig = [{'linear_x': 0.2, 'angular_z': s, 'duration': 1} for s in (0.3, -0.3, 0.3, -0.3)]
    assert turn_direction_changes(zig) == 3


# ----------------------------------------------------------------- analysis

def _record(group, case_id, mode='single', **fields):
    base = {'group': group, 'case_id': case_id, 'mode': mode, 'error_type': '',
            'inference_time_sec': 1.0}
    base.update(fields)
    return base


def test_relations_agree_disagree_tie_and_missing():
    from experiment_cases import A1
    summaries = [{'case_id': cid, 'stats': {'abs_linear_x': {'mean': mean}}}
                 for cid, mean in (('A1-02', 0.1), ('A1-03', 0.1), ('A1-06', 0.3),
                                   ('A1-07', 0.2))]
    results, agreement = analyze_results.evaluate_relations(A1, summaries)
    outcomes = {(r['a'], r['b']): r['outcome'] for r in results}
    assert outcomes[('A1-02', 'A1-03')] == 'tie'          # strict '<' with equal means
    assert outcomes[('A1-03', 'A1-06')] == 'agree'
    assert outcomes[('A1-06', 'A1-07')] == 'disagree'
    assert outcomes[('A1-07', 'A1-08')] == 'not_evaluated'
    evaluated = [r for r in results if r['outcome'] != 'not_evaluated']
    assert agreement == pytest.approx(
        sum(r['outcome'] == 'agree' for r in evaluated) / len(evaluated))


def test_identical_output_ratio_and_indefinite_classes():
    from experiment_cases import C
    case = C.cases[8]                                       # 앞으로 계속 이동해
    rec = [evaluate_single(_twist(V, 0, T), LIMITS),
           evaluate_single(_twist(V, 0, T), LIMITS),
           evaluate_single(_twist(0, 0, 0), LIMITS),
           evaluate_single(_twist(V, 0, 6 * T), LIMITS)]
    records = [_record('C', case.case_id, **r, inference_time_sec=0.5) for r in rec]
    summary = analyze_results.summarize_case(C, case, records, LIMITS)
    assert summary['identical_output_ratio'] == pytest.approx(0.5)
    assert summary['duration_classes'] == {'finite': 2, 'refused': 1, 'over_limit': 1}
    assert summary['validation_pass_rate'] == pytest.approx(0.75)
    assert summary['stats']['duration']['max'] == 6 * T     # rejected value still counted


# ------------------------------------------------------------ runner (mock)

def test_mock_run_all_writes_csv_json_and_plots(tmp_path, capsys):
    code = run_llm_experiment.main(['--experiment', 'all', '--repeat', '2', '--backend', 'mock',
                                    '--output-dir', str(tmp_path), '--run-id', 'm', '--quiet'])
    assert code == 0
    run_dir = tmp_path / 'm'
    for name in ('raw.csv', 'raw.json', 'metadata.json', 'summary.csv', 'summary.json',
                 'report.md'):
        assert (run_dir / name).is_file(), name
    with open(run_dir / 'raw.csv', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60 * 2
    assert tuple(rows[0]) == run_llm_experiment.CSV_FIELDS
    data = json.loads((run_dir / 'raw.json').read_text(encoding='utf-8'))
    d2 = [r for r in data['records'] if r['group'] == 'D2' and r['json_valid']]
    assert d2 and all(r['path'] and r['actions'] is not None for r in d2)
    assert data['metadata']['model'].startswith('mock')
    assert 'mock' in (run_dir / 'report.md').read_text(encoding='utf-8')
    pytest.importorskip('matplotlib')
    assert len(list((run_dir / 'plots').glob('*.png'))) == 7


def test_case_filter_and_errors_do_not_stop_the_run(tmp_path, monkeypatch):
    class Flaky:
        calls = 0

        def ask(self, system_prompt, schema, command):
            Flaky.calls += 1
            if Flaky.calls % 2:
                raise LLMTimeoutError('Ollama did not answer within 1.0 s')
            return '{"linear_x": 0.1, "angular_z": 0, "duration": 1}'

        def server_timing(self):
            return {}

    monkeypatch.setattr(run_llm_experiment, 'make_client', lambda args: Flaky())
    code = run_llm_experiment.main(['--experiment', 'B', '--repeat', '3', '--cases', 'B-01,B-02',
                                    '--output-dir', str(tmp_path), '--run-id', 'f', '--quiet',
                                    '--no-analyze'])
    assert code == 0
    records = json.loads((tmp_path / 'f' / 'raw.json').read_text(encoding='utf-8'))['records']
    assert len(records) == 6
    assert {r['case_id'] for r in records} == {'B-01', 'B-02'}
    timeouts = [r for r in records if r['error_type'] == 'LLMTimeoutError']
    assert len(timeouts) == 3 and all(not r['json_valid'] for r in timeouts)


def test_unreachable_server_exits_with_clear_message(tmp_path, capsys):
    code = run_llm_experiment.main(['--experiment', 'A', '--repeat', '1',
                                    '--api-url', 'http://127.0.0.1:9/api/chat',
                                    '--output-dir', str(tmp_path), '--run-id', 'x'])
    assert code == 2
    assert '연결할 수 없습니다' in capsys.readouterr().err
    assert not (tmp_path / 'x').exists()


def test_preflight_reports_missing_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({'models': [{'name': 'llama3:8b'}]}).encode()

    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: Response())
    with pytest.raises(Exception, match='ollama pull qwen3:4b'):
        preflight('http://localhost:11434/api/chat', 'qwen3:4b')
    with pytest.raises(LLMConnectionError):
        monkeypatch.undo()
        preflight('http://127.0.0.1:9/api/chat', 'qwen3:4b', timeout_sec=0.5)
