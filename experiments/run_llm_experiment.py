#!/usr/bin/env python3
"""Run the natural-language motion inference experiments (no ROS, no robot).

    python3 run_llm_experiment.py --experiment all --repeat 10
    python3 run_llm_experiment.py --experiment A --repeat 10
    python3 run_llm_experiment.py --experiment D --repeat 3 --backend mock

Each run writes results/<run_id>/{raw.csv, raw.json, metadata.json} and, unless
--no-analyze is given, the analysis outputs (summary, report, plots).
Nothing is published to ROS; the LLM output is only scored.
"""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
# Run straight from the source tree: no colcon build or sourcing needed.
for path in (str(PACKAGE_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiment_cases import groups_for  # noqa: E402
from experiment_clients import ExperimentOllamaClient  # noqa: E402
from experiment_clients import MockExperimentClient  # noqa: E402
from experiment_clients import preflight  # noqa: E402
from experiment_core import evaluate  # noqa: E402
from experiment_prompts import build_prompts  # noqa: E402
from experiment_prompts import current_limits  # noqa: E402
from experiment_prompts import PROMPT_VARIANTS  # noqa: E402
from llm_velocity_controller.llm_client import LLMError  # noqa: E402

DEFAULT_MODEL = 'qwen3:4b'
DEFAULT_API_URL = 'http://localhost:11434/api/chat'
DEFAULT_RESULTS_DIR = HERE / 'results'

CSV_FIELDS = (
    'run_id', 'experiment', 'group', 'mode', 'case_id', 'command', 'repeat',
    'linear_x', 'angular_z', 'duration',
    'estimated_distance', 'estimated_rotation_rad', 'estimated_rotation_deg', 'radius',
    'n_actions', 'path_length', 'closure_distance',
    'inference_time_sec', 'json_valid', 'validation_passed', 'validation_error',
    'refused', 'limit_violations', 'error_type', 'error', 'raw_response',
)
# Paths are kept in raw.json only where the geometry is the point of the test.
PATH_GROUPS = ('D1', 'D2')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', default='all', choices=('A', 'B', 'C', 'D', 'all'))
    parser.add_argument('--repeat', type=int, default=10, help='repetitions per command')
    parser.add_argument('--backend', default='ollama', choices=('ollama', 'mock'))
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--api-url', default=DEFAULT_API_URL)
    parser.add_argument('--timeout', type=float, default=60.0, help='seconds per request')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=None, help='Ollama sampling seed')
    parser.add_argument('--keep-alive', default='30m')
    parser.add_argument('--prompt-variant', default='with_limits', choices=PROMPT_VARIANTS,
                        help='with_limits tells the model the robot limits; no_limits does not')
    parser.add_argument('--cases', default='',
                        help='comma-separated case ids to run, e.g. A1-02,B-04')
    parser.add_argument('--limit-cases', type=int, default=0,
                        help='run only the first N cases of each group (smoke tests)')
    parser.add_argument('--output-dir', default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument('--run-id', default='', help='default: timestamp')
    parser.add_argument('--no-analyze', action='store_true')
    parser.add_argument('--mock-error-rate', type=float, default=0.05)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error('--repeat must be at least 1')
    return args


def _git_info():
    def git(*cmd):
        try:
            return subprocess.run(['git', '-C', str(PACKAGE_ROOT), *cmd], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ''
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'status_short': git('status', '--short')}


def select_cases(groups, case_filter, limit_cases):
    wanted = {c.strip() for c in case_filter.split(',') if c.strip()}
    plan = []
    for group in groups:
        cases = [c for c in group.cases if not wanted or c.case_id in wanted]
        if limit_cases > 0:
            cases = cases[:limit_cases]
        plan += [(group, case) for case in cases]
    if wanted:
        known = {case.case_id for group in groups for case in group.cases}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f'알 수 없는 case id: {", ".join(unknown)}')
    return plan


def make_client(args):
    if args.backend == 'mock':
        return MockExperimentClient(seed=args.seed or 0, error_rate=args.mock_error_rate)
    preflight(args.api_url, args.model)
    return ExperimentOllamaClient(model=args.model, api_url=args.api_url,
                                  timeout_sec=args.timeout, keep_alive=args.keep_alive,
                                  temperature=args.temperature, seed=args.seed)


def run_one(client, group, case, repeat, prompts, limits, run_id):
    system_prompt, schema = prompts[group.prompt_key]
    record = {'run_id': run_id, 'experiment': group.experiment, 'group': group.group_id,
              'mode': group.mode, 'case_id': case.case_id, 'command': case.command,
              'repeat': repeat, 'error_type': '', 'error': '', 'raw_response': ''}
    started = time.perf_counter()
    try:
        raw = client.ask(system_prompt, schema, case.command)
    except Exception as error:  # connection, timeout, bad envelope, anything: keep going
        record['inference_time_sec'] = time.perf_counter() - started
        record.update(error_type=type(error).__name__, error=str(error))
        record.update(evaluate('', group.mode, limits))
        record['validation_error'] = f'no response: {error}'
        record['server_timing'] = {}
        return record
    record['inference_time_sec'] = time.perf_counter() - started
    record['raw_response'] = raw
    record['server_timing'] = client.server_timing()
    record.update(evaluate(raw, group.mode, limits))
    return record


def _csv_row(record):
    row = {}
    for name in CSV_FIELDS:
        value = record.get(name)
        if isinstance(value, list):
            value = ';'.join(map(str, value))
        row[name] = '' if value is None else value
    return row


def _format_progress(index, total, record):
    head = f'[{index}/{total}] {record["case_id"]} #{record["repeat"]} {record["command"]}'
    if record['error_type']:
        return f'{head}\n    오류 {record["error_type"]}: {record["error"]}'
    if not record['json_valid']:
        return f'{head}\n    JSON 무효: {record["validation_error"]}'

    def fmt(name):
        value = record[name]
        return '-' if value is None else f'{value:.3g}'
    status = '통과' if record['validation_passed'] else f'거부({record["validation_error"]})'
    shape = f'actions={record["n_actions"]} ' if record['mode'] == 'actions' else ''
    return (f'{head}\n    {shape}v={fmt("linear_x")} w={fmt("angular_z")} '
            f'T={fmt("duration")} d={fmt("estimated_distance")} '
            f'rot={fmt("estimated_rotation_deg")}° '
            f'({record["inference_time_sec"]:.2f}s) {status}')


def run(args):
    run_id = args.run_id or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    groups = groups_for(args.experiment)
    plan = select_cases(groups, args.cases, args.limit_cases)
    prompts = build_prompts(args.prompt_variant)
    limits = current_limits()

    try:
        client = make_client(args)
    except LLMError as error:
        run_dir.rmdir()
        print(f'[실행 불가] {error}', file=sys.stderr)
        return 2

    metadata = {
        'run_id': run_id,
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'argv': sys.argv[1:],
        'backend': args.backend,
        'model': args.model if args.backend == 'ollama' else 'mock (NOT an LLM result)',
        'api_url': args.api_url,
        'options': {'temperature': args.temperature, 'seed': args.seed, 'think': False,
                    'stream': False, 'timeout_sec': args.timeout, 'keep_alive': args.keep_alive},
        'repeat': args.repeat,
        'prompt_variant': args.prompt_variant,
        'limits': limits,
        'groups': sorted({g.group_id for g, _ in plan}),
        'n_cases': len(plan),
        'prompts': {key: {'system_prompt': p, 'schema': s} for key, (p, s) in prompts.items()},
        'git': _git_info(),
    }
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    total = len(plan) * args.repeat
    print(f'실행 ID {run_id}: {len(plan)}개 명령 × {args.repeat}회 = {total}회 추론 '
          f'({metadata["model"]}, temperature={args.temperature}, 결과: {run_dir})')
    print(f'검증 한계(validator.py): {limits}')

    records = []
    interrupted = False
    with open(run_dir / 'raw.csv', 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        index = 0
        try:
            for group, case in plan:
                for repeat in range(1, args.repeat + 1):
                    index += 1
                    record = run_one(client, group, case, repeat, prompts, limits, run_id)
                    if group.group_id not in PATH_GROUPS:
                        record['path'] = None
                    records.append(record)
                    writer.writerow(_csv_row(record))
                    stream.flush()            # partial results survive a crash
                    if not args.quiet:
                        print(_format_progress(index, total, record), flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print('\n중단됨: 지금까지의 결과를 저장합니다.', file=sys.stderr)

    metadata.update(finished_at=datetime.datetime.now().isoformat(timespec='seconds'),
                    completed_inferences=len(records), planned_inferences=total,
                    interrupted=interrupted)
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                           encoding='utf-8')
    (run_dir / 'raw.json').write_text(
        json.dumps({'metadata': metadata, 'records': records}, ensure_ascii=False, indent=1),
        encoding='utf-8')

    errors = sum(1 for r in records if r['error_type'])
    print(f'완료: {len(records)}/{total}회, 통신 오류 {errors}회 → {run_dir}')
    if args.backend == 'ollama' and records and errors == len(records):
        print('[경고] 모든 요청이 실패했습니다. Ollama 서버 상태와 --timeout을 확인하세요.',
              file=sys.stderr)
    if not args.no_analyze and records:
        from analyze_results import analyze
        analyze(run_dir)
    return 130 if interrupted else 0


def main(argv=None):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    return run(parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
