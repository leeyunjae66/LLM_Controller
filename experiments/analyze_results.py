#!/usr/bin/env python3
"""Statistics, semantic-order agreement and plots for one experiment run.

    python3 analyze_results.py results/<run_id>

Writes summary.csv, summary.json, report.md and plots/*.png into the run
directory. Uses only the standard library plus matplotlib (optional): if
matplotlib is missing, the numbers are still produced and plotting is skipped.

Value statistics use every response that is well-formed JSON with numeric
fields, including ones the validator rejected (an out-of-range choice is part
of what is measured). Validator pass rate is reported separately.
"""

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys
import warnings

HERE = Path(__file__).resolve().parent
for path in (str(HERE.parent), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiment_cases import group_by_id  # noqa: E402
from experiment_core import classify_segments  # noqa: E402
from experiment_core import turn_direction_changes  # noqa: E402

# ---- evaluation-only tolerances (never shown to the model) -------------------
UTURN_TOLERANCE_DEG = 20.0          # |rotation| within 180 ± this
FULL_CIRCLE_DEG = 330.0             # "drew a full circle"
CORNER_TOLERANCE_DEG = 15.0         # rectangle/square corners within 90 ± this
CLOSURE_FRACTION = 0.15             # closed shape: end within 15 % of path length of start
SQUARE_SIDE_CV = 0.10               # square: side length coefficient of variation
RECTANGLE_SIDE_RATIO = 1.2          # rectangle: longest / shortest side
IN_PLACE_PATH_M = 0.05              # "in place": path length at most this

METRICS = ('linear_x', 'angular_z', 'abs_linear_x', 'abs_angular_z', 'duration',
           'distance', 'rotation_deg', 'radius', 'path_length', 'closure_distance',
           'n_actions')


def metric_value(record, metric):
    if not record.get('json_valid'):
        return None
    if metric == 'abs_linear_x':
        v = record.get('linear_x')
        return None if v is None else abs(v)
    if metric == 'abs_angular_z':
        w = record.get('angular_z')
        return None if w is None else abs(w)
    key = {'distance': 'estimated_distance', 'rotation_deg': 'estimated_rotation_deg'}.get(
        metric, metric)
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'n': 0, 'mean': None, 'std': None, 'min': None, 'max': None}
    return {'n': len(values), 'mean': statistics.fmean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values)}


def output_key(record):
    """What counts as 'the same output' across repeats."""
    if record.get('error_type'):
        return ('error', record['error_type'])
    if not record.get('json_valid'):
        return ('invalid', record.get('validation_error', ''))
    if record['mode'] == 'actions':
        return ('actions', tuple((round(a['linear_x'], 6), round(a['angular_z'], 6),
                                  round(a['duration'], 6)) for a in record['actions']))
    return ('single', round(record['linear_x'], 6), round(record['angular_z'], 6),
            round(record['duration'], 6))


def _rate(flags):
    flags = list(flags)
    return sum(1 for f in flags if f) / len(flags) if flags else None


def _sign(value):
    if value is None:
        return None
    return 0 if abs(value) < 1e-9 else (1 if value > 0 else -1)


def duration_class(record, duration_max):
    """For open-ended commands: how did the model bound the motion?"""
    if record.get('error_type'):
        return 'error'
    if not record.get('json_valid'):
        return 'invalid'
    if record.get('refused'):
        return 'refused'
    duration = record.get('duration')
    if duration is not None and duration > duration_max:
        return 'over_limit'
    return 'finite'


def shape_checks(case, record):
    """Heuristic geometry checks for experiment D. Returns {check: bool}."""
    if not record.get('json_valid') or record.get('refused'):
        return {}
    checks = {}
    rotation = record.get('estimated_rotation_deg') or 0.0
    path_length = record.get('path_length') or 0.0
    tags = set(case.tags)
    if 'uturn' in tags:
        checks['uturn_angle'] = abs(abs(rotation) - 180.0) <= UTURN_TOLERANCE_DEG
        if 'in_place' in tags:
            checks['in_place'] = path_length <= IN_PLACE_PATH_M
        if 'moving' in tags:
            checks['moves_forward'] = (record.get('estimated_distance') or 0.0) > 0.0
    if 'circle' in tags:
        checks['full_circle'] = abs(rotation) >= FULL_CIRCLE_DEG
        if case.angular_sign:
            checks['turn_direction'] = _sign(rotation) == case.angular_sign
    if record['mode'] != 'actions':
        return checks
    straights, turns, _ = classify_segments(record.get('actions'))
    if tags & {'rectangle', 'square'}:
        corners = [t for t in turns if abs(abs(t) - 90.0) <= CORNER_TOLERANCE_DEG]
        checks['four_sides'] = len(straights) >= 4
        checks['right_angle_corners'] = len(corners) >= 3
        checks['closed'] = (path_length > 0.0 and record['closure_distance']
                            <= CLOSURE_FRACTION * path_length)
        sides = straights[:4]
        if len(sides) == 4 and min(sides) > 0.0:
            if 'square' in tags:
                cv = statistics.pstdev(sides) / statistics.fmean(sides)
                checks['equal_sides'] = cv <= SQUARE_SIDE_CV
            else:
                checks['unequal_sides'] = max(sides) / min(sides) >= RECTANGLE_SIDE_RATIO
    if 'zigzag' in tags:
        checks['alternating_turns'] = turn_direction_changes(record.get('actions')) >= 2
        checks['moves_forward'] = (record.get('estimated_distance') or 0.0) > 0.0
    return checks


def summarize_case(group, case, records, limits):
    summary = {
        'group': group.group_id, 'experiment': group.experiment, 'mode': group.mode,
        'case_id': case.case_id, 'command': case.command, 'note': case.note,
        'tags': list(case.tags), 'n_runs': len(records),
        'error_rate': _rate(r['error_type'] for r in records),
        'json_valid_rate': _rate(r['json_valid'] for r in records),
        'validation_pass_rate': _rate(r['validation_passed'] for r in records),
        'refused_rate': _rate(bool(r.get('refused')) for r in records),
        'limit_violation_rate': _rate(bool(r.get('limit_violations')) for r in records),
        'mean_inference_time_sec': statistics.fmean(r['inference_time_sec'] for r in records)
        if records else None,
    }
    keys = Counter(output_key(r) for r in records)
    summary['identical_output_ratio'] = (keys.most_common(1)[0][1] / len(records)
                                         if records else None)
    summary['distinct_outputs'] = len(keys)
    summary['stats'] = {m: describe(metric_value(r, m) for r in records) for m in METRICS}

    valid = [r for r in records if r.get('json_valid')]
    if case.linear_sign is not None:
        summary['linear_sign_accuracy'] = _rate(_sign(r['linear_x']) == case.linear_sign
                                                for r in valid if r.get('linear_x') is not None)
    if case.angular_sign is not None:
        summary['angular_sign_accuracy'] = _rate(
            _sign(r['angular_z']) == case.angular_sign
            for r in valid if r.get('angular_z') is not None)
    if case.reference_range is not None:
        low, high = case.reference_range
        values = [metric_value(r, group.main_metric) for r in valid]
        summary['reference_range'] = [low, high]
        summary['in_reference_range_rate'] = _rate(
            low <= v <= high for v in values if v is not None)
    if group.experiment == 'C':
        summary['duration_classes'] = dict(Counter(duration_class(r, limits['duration_max'])
                                                   for r in records))
    if group.experiment == 'D':
        per_record = [shape_checks(case, r) for r in records]
        names = sorted({name for checks in per_record for name in checks})
        summary['shape_checks'] = {
            name: _rate(checks.get(name, False) for checks in per_record if checks)
            for name in names}
        summary['shape_checked_runs'] = sum(1 for checks in per_record if checks)
    return summary


def evaluate_relations(group, case_summaries):
    by_id = {s['case_id']: s for s in case_summaries}
    results = []
    for rel in group.relations:
        a, b = by_id.get(rel.a), by_id.get(rel.b)
        mean_a = a['stats'][rel.metric]['mean'] if a else None
        mean_b = b['stats'][rel.metric]['mean'] if b else None
        if mean_a is None or mean_b is None:
            outcome = 'not_evaluated'
        elif math.isclose(mean_a, mean_b, rel_tol=1e-9, abs_tol=1e-12):
            outcome = 'agree' if rel.op == '<=' else 'tie'
        else:
            outcome = 'agree' if mean_a < mean_b else 'disagree'
        results.append({'metric': rel.metric, 'a': rel.a, 'op': rel.op, 'b': rel.b,
                        'mean_a': mean_a, 'mean_b': mean_b, 'outcome': outcome})
    evaluated = [r for r in results if r['outcome'] != 'not_evaluated']
    agreement = (sum(1 for r in evaluated if r['outcome'] == 'agree') / len(evaluated)
                 if evaluated else None)
    return results, agreement


def summarize_group(group, case_summaries, records):
    relations, agreement = evaluate_relations(group, case_summaries)
    out = {
        'group': group.group_id, 'title': group.title, 'experiment': group.experiment,
        'mode': group.mode, 'main_metric': group.main_metric, 'n_runs': len(records),
        'n_cases': len(case_summaries),
        'json_valid_rate': _rate(r['json_valid'] for r in records),
        'validation_pass_rate': _rate(r['validation_passed'] for r in records),
        'error_rate': _rate(r['error_type'] for r in records),
        'mean_inference_time_sec': statistics.fmean(r['inference_time_sec'] for r in records)
        if records else None,
        'mean_identical_output_ratio': statistics.fmean(
            s['identical_output_ratio'] for s in case_summaries) if case_summaries else None,
        'order_agreement': agreement,
        'relations': relations,
    }
    by_id = {s['case_id']: s for s in case_summaries}
    mirrors = []
    for a, b in group.expectations.get('mirror_pairs', ()):
        ma = by_id.get(a, {}).get('stats', {}).get('abs_angular_z', {}).get('mean')
        mb = by_id.get(b, {}).get('stats', {}).get('abs_angular_z', {}).get('mean')
        if ma is not None and mb is not None:
            mirrors.append({'a': a, 'b': b, 'abs_a': ma, 'abs_b': mb, 'abs_diff': abs(ma - mb)})
    if mirrors:
        out['mirror_pairs'] = mirrors
    ratios = []
    for a, b, expected in group.expectations.get('ratio_pairs', ()):
        ma = by_id.get(a, {}).get('stats', {}).get(group.main_metric, {}).get('mean')
        mb = by_id.get(b, {}).get('stats', {}).get(group.main_metric, {}).get('mean')
        if ma and mb is not None:
            ratios.append({'a': a, 'b': b, 'expected_ratio': expected, 'ratio': mb / ma})
    if ratios:
        out['ratio_pairs'] = ratios
    return out


def analyze_records(records, limits):
    groups = []
    cases = []
    for group_id in sorted({r['group'] for r in records}):
        group = group_by_id(group_id)
        group_records = [r for r in records if r['group'] == group_id]
        case_summaries = []
        for case in group.cases:
            case_records = [r for r in group_records if r['case_id'] == case.case_id]
            if case_records:
                case_summaries.append(summarize_case(group, case, case_records, limits))
        cases += case_summaries
        groups.append(summarize_group(group, case_summaries, group_records))
    return {'groups': groups, 'cases': cases}


# ------------------------------------------------------------------ outputs

def _fmt(value, digits=3):
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.{digits}g}'
    return str(value)


def _pct(value):
    return '-' if value is None else f'{100 * value:.0f}%'


SUMMARY_CSV_FIELDS = ('group', 'case_id', 'command', 'n_runs', 'json_valid_rate',
                      'validation_pass_rate', 'refused_rate', 'error_rate',
                      'identical_output_ratio', 'distinct_outputs', 'mean_inference_time_sec')


def write_summary_csv(path, cases):
    stat_fields = [f'{m}_{s}' for m in METRICS for s in ('mean', 'std', 'min', 'max')]
    extra = ('linear_sign_accuracy', 'angular_sign_accuracy', 'in_reference_range_rate')
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SUMMARY_CSV_FIELDS) + list(extra)
                                + stat_fields + ['note'])
        writer.writeheader()
        for case in cases:
            row = {k: case.get(k) for k in SUMMARY_CSV_FIELDS + extra}
            for m in METRICS:
                for s in ('mean', 'std', 'min', 'max'):
                    row[f'{m}_{s}'] = case['stats'][m][s]
            row['note'] = case['note']
            writer.writerow({k: '' if v is None else v for k, v in row.items()})


def write_report(path, metadata, analysis):
    limits = metadata.get('limits', {})
    lines = [f'# LLM 운동 추론 실험 결과 `{metadata.get("run_id")}`', '']
    if metadata.get('backend') == 'mock':
        lines += ['> ⚠️ **mock 백엔드 결과입니다. LLM 추론 결과가 아니며 파이프라인 확인용입니다.**', '']
    lines += [
        f'- 모델: `{metadata.get("model")}`, '
        f'temperature={metadata.get("options", {}).get("temperature")}, '
        f'반복 {metadata.get("repeat")}회, 프롬프트 `{metadata.get("prompt_variant")}`',
        f'- 검증 한계: |linear_x| ≤ {limits.get("linear_x_limit")} m/s, '
        f'|angular_z| ≤ {limits.get("angular_z_limit")} rad/s, '
        f'duration ≤ {limits.get("duration_max")} s',
        f'- 추론 {metadata.get("completed_inferences")}/{metadata.get("planned_inferences")}회'
        + (' (중단됨)' if metadata.get('interrupted') else ''),
    ]
    status = metadata.get('git', {}).get('status_short')
    if status:
        lines.append(f'- 작업 트리에 커밋되지 않은 변경 있음: `{status.replace(chr(10), "; ")}`')
    lines.append('')
    main_label = {'abs_linear_x': '|linear_x| (m/s)', 'abs_angular_z': '|angular_z| (rad/s)',
                  'distance': '추정 거리 (m)', 'duration': 'duration (s)',
                  'rotation_deg': '회전각 (deg)'}
    for group in analysis['groups']:
        metric = group['main_metric']
        lines += [f'## {group["group"]}: {group["title"]}', '',
                  f'JSON 유효 {_pct(group["json_valid_rate"])}, Validator 통과 '
                  f'{_pct(group["validation_pass_rate"])}, 통신 오류 {_pct(group["error_rate"])}, '
                  f'평균 추론 {_fmt(group["mean_inference_time_sec"])} s, '
                  f'평균 동일 출력 비율 {_pct(group["mean_identical_output_ratio"])}, '
                  f'**의미 순서 일치율 {_pct(group["order_agreement"])}**', '',
                  f'| case | 명령 | 통과 | {main_label.get(metric, metric)} 평균±표준편차 | 최소~최대 '
                  '| linear_x | angular_z | duration | 동일 출력 | 추론(s) |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for case in (c for c in analysis['cases'] if c['group'] == group['group']):
            st = case['stats']
            main = st[metric]
            lines.append(
                f'| {case["case_id"]} | {case["command"]} | {_pct(case["validation_pass_rate"])} '
                f'| {_fmt(main["mean"])} ± {_fmt(main["std"])} '
                f'| {_fmt(main["min"])}~{_fmt(main["max"])} '
                f'| {_fmt(st["linear_x"]["mean"])} | {_fmt(st["angular_z"]["mean"])} '
                f'| {_fmt(st["duration"]["mean"])} | {_pct(case["identical_output_ratio"])} '
                f'| {_fmt(case["mean_inference_time_sec"])} |')
        lines.append('')
        if group['relations']:
            lines += ['의미 순서 비교 (평균 기준):', '']
            for rel in group['relations']:
                mark = {'agree': '✅', 'disagree': '❌', 'tie': '⚖️ 동률',
                        'not_evaluated': '— 평가 불가'}[rel['outcome']]
                lines.append(f'- {rel["a"]} {rel["op"]} {rel["b"]} ({rel["metric"]}): '
                             f'{_fmt(rel["mean_a"])} vs {_fmt(rel["mean_b"])} → {mark}')
            lines.append('')
        extras = []
        for case in (c for c in analysis['cases'] if c['group'] == group['group']):
            parts = []
            for key, label in (('linear_sign_accuracy', '직진 부호'),
                               ('angular_sign_accuracy', '회전 부호'),
                               ('in_reference_range_rate', '기준 범위 내')):
                if case.get(key) is not None:
                    parts.append(f'{label} {_pct(case[key])}')
            if case.get('reference_range'):
                parts.append(f'기준 {case["reference_range"][0]}~{case["reference_range"][1]} m')
            if case.get('duration_classes'):
                parts.append('종료 처리 ' + ', '.join(
                    f'{k} {v}' for k, v in case['duration_classes'].items()))
            if case.get('shape_checks'):
                parts.append('형상 ' + ', '.join(
                    f'{k} {_pct(v)}' for k, v in case['shape_checks'].items()))
            if case.get('note'):
                parts.append(f'메모: {case["note"]}')
            if parts:
                extras.append(f'- {case["case_id"]}: ' + ' / '.join(parts))
        for m in group.get('mirror_pairs', ()):
            extras.append(f'- 좌우 대칭 {m["a"]} vs {m["b"]}: |w| {_fmt(m["abs_a"])} vs '
                          f'{_fmt(m["abs_b"])} (차이 {_fmt(m["abs_diff"])})')
        for r in group.get('ratio_pairs', ()):
            extras.append(f'- 배수 {r["b"]}/{r["a"]}: {_fmt(r["ratio"])} (기대 {r["expected_ratio"]})')
        if extras:
            lines += ['세부 지표:', ''] + extras + ['']
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ------------------------------------------------------------------- plots

def _has_hangul(font_file):
    try:
        from matplotlib.ft2font import FT2Font
        return FT2Font(font_file).get_char_index(ord('가')) != 0
    except Exception:  # unreadable font file: just treat it as unusable
        return False


def _setup_matplotlib():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        return None, False
    korean = False
    # Noto CJK .ttc collections register under their first face ("... JP"),
    # which still contains Hangul, so check glyph coverage rather than names.
    preferred = ('Noto Sans CJK KR', 'NanumGothic', 'Noto Sans KR', 'Noto Sans CJK JP',
                 'UnDotum', 'Baekmuk Gulim')
    fonts = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in fonts and _has_hangul(fonts[name]):
            plt.rcParams['font.family'] = name
            korean = True
            break
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', message='Glyph .* missing')
    return plt, korean


def _labels(cases, korean):
    if korean:
        return [f'{c["case_id"]}\n{c["command"]}' for c in cases]
    return [c['case_id'] for c in cases]


def _bar_plot(plt, cases, metric, ylabel, title, path, korean, ref_ranges=False, hline=None,
              signed=False):
    if not cases:
        return None
    means = [c['stats'][metric]['mean'] or 0.0 for c in cases]
    stds = [c['stats'][metric]['std'] or 0.0 for c in cases]
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(cases)), 5))
    colors = ['tab:red' if signed and m < 0 else 'tab:blue' for m in means]
    xs = range(len(cases))
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, alpha=0.8)
    if ref_ranges:
        for x, case in zip(xs, cases):
            if case.get('reference_range'):
                low, high = case['reference_range']
                ax.add_patch(plt.Rectangle((x - 0.45, low), 0.9, high - low, fill=False,
                                           edgecolor='green', linestyle='--', linewidth=1.5))
        ax.plot([], [], color='green', linestyle='--', label='reference range (eval only)')
        ax.legend()
    if hline is not None:
        ax.axhline(hline, color='gray', linestyle=':', label=f'limit {hline}')
        ax.legend()
    if signed:
        ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(_labels(cases, korean), rotation=45 if korean else 0,
                       ha='right' if korean else 'center', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _distribution_plot(plt, analysis, records, path, korean):
    panels = [(g, g['main_metric']) for g in analysis['groups']
              if g['group'] in ('A1', 'A2', 'B', 'C')]
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)), squeeze=False)
    for ax, (group, metric) in zip(axes[:, 0], panels):
        cases = [c for c in analysis['cases'] if c['group'] == group['group']]
        for x, case in enumerate(cases):
            values = [metric_value(r, metric) for r in records if r['case_id'] == case['case_id']
                      and r['group'] == group['group']]
            values = [v for v in values if v is not None]
            jitter = [x + 0.08 * ((i % 5) - 2) for i in range(len(values))]
            ax.scatter(jitter, values, s=18, alpha=0.6)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([c['case_id'] for c in cases], fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f'{group["group"]} repeat distribution')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _paths_plot(plt, analysis, records, group_id, path, korean):
    cases = [c for c in analysis['cases'] if c['group'] == group_id]
    if not cases:
        return None
    cols = 5
    rows = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, case in zip(axes.flat, cases):
        ax.set_visible(True)
        drawn = 0
        for r in records:
            if r['group'] == group_id and r['case_id'] == case['case_id'] and r.get('path'):
                xs = [p[1] for p in r['path']]
                ys = [p[2] for p in r['path']]
                ax.plot(xs, ys, alpha=0.5, linewidth=1.2)
                drawn += 1
        ax.plot(0, 0, 'ko', markersize=4)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)
        title = f'{case["case_id"]} {case["command"]}' if korean else case['case_id']
        ax.set_title(title if drawn else f'{title}\n(no motion)', fontsize=8)
    fig.suptitle(f'{group_id} ideal XY paths (start at origin facing +x)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def make_plots(run_dir, analysis, records, metadata):
    plt, korean = _setup_matplotlib()
    if plt is None:
        print('matplotlib이 없어 그래프를 건너뜁니다.')
        return []
    plots = Path(run_dir) / 'plots'
    plots.mkdir(exist_ok=True)

    def group_cases(group_id):
        return [c for c in analysis['cases'] if c['group'] == group_id]
    made = [
        _bar_plot(plt, group_cases('A1'), 'abs_linear_x', '|linear_x| (m/s)',
                  'A1 linear speed by adverb (mean ± std)', plots / 'a1_linear_speed.png', korean,
                  hline=metadata.get('limits', {}).get('linear_x_limit')),
        _bar_plot(plt, group_cases('A2'), 'angular_z', 'angular_z (rad/s)',
                  'A2 yaw rate by adverb (mean ± std, sign = direction)',
                  plots / 'a2_angular_speed.png', korean, signed=True),
        _bar_plot(plt, group_cases('B'), 'distance', 'linear_x * duration (m)',
                  'B estimated distance (mean ± std)', plots / 'b_distance.png', korean,
                  ref_ranges=True),
        _bar_plot(plt, group_cases('C'), 'duration', 'duration (s)',
                  'C duration by time expression (mean ± std)', plots / 'c_duration.png', korean,
                  hline=metadata.get('limits', {}).get('duration_max')),
        _distribution_plot(plt, analysis, records, plots / 'repeat_distribution.png', korean),
        _paths_plot(plt, analysis, records, 'D1', plots / 'd1_paths_single.png', korean),
        _paths_plot(plt, analysis, records, 'D2', plots / 'd2_paths_actions.png', korean),
    ]
    return [str(p) for p in made if p]


def analyze(run_dir):
    run_dir = Path(run_dir)
    data = json.loads((run_dir / 'raw.json').read_text(encoding='utf-8'))
    metadata, records = data['metadata'], data['records']
    analysis = analyze_records(records, metadata['limits'])
    write_summary_csv(run_dir / 'summary.csv', analysis['cases'])
    plots = make_plots(run_dir, analysis, records, metadata)
    analysis['plots'] = plots
    (run_dir / 'summary.json').write_text(
        json.dumps({'metadata': metadata, **analysis}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    write_report(run_dir / 'report.md', metadata, analysis)
    print(f'분석 완료: {run_dir / "report.md"} (그래프 {len(plots)}개)')
    for group in analysis['groups']:
        print(f'  {group["group"]:<3} 통과율 {_pct(group["validation_pass_rate"]):>4}  '
              f'순서 일치율 {_pct(group["order_agreement"]):>4}  '
              f'평균 추론 {_fmt(group["mean_inference_time_sec"])} s')
    return analysis


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print('사용법: python3 analyze_results.py results/<run_id>', file=sys.stderr)
        return 2
    analyze(argv[0])
    return 0


if __name__ == '__main__':
    sys.exit(main())
