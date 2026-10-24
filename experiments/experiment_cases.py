"""Natural-language test commands and the evaluation criteria applied to them.

Everything in this file is used only *after* inference, to score the outputs.
None of it is sent to the LLM: prompts live in experiment_prompts.py and must
not contain these commands, orderings or reference ranges.

Relations compare the mean of a metric between two cases of the same
experiment group. Only pairs whose order follows from the wording are listed;
expressions that are not comparable (e.g. 살살 vs 느긋하게) get no relation, so
no arbitrary rank is invented for them.
"""

from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class Case:
    case_id: str
    command: str
    # Expected sign of the mean output: +1, -1, 0 (must be zero) or None (no check).
    linear_sign: int = None
    angular_sign: int = None
    # Inclusive [low, high] reference interval for the group's main metric, or None.
    reference_range: tuple = None
    # Free-text note for ambiguous wording (recorded, never scored automatically).
    note: str = ''
    tags: tuple = ()


@dataclass(frozen=True)
class Relation:
    """mean(metric of a) <op> mean(metric of b), op in {'<', '<='}."""
    metric: str
    a: str
    op: str
    b: str


@dataclass(frozen=True)
class Group:
    group_id: str
    title: str
    experiment: str          # A, B, C or D
    mode: str                # 'single' (one Twist) or 'actions' (sequence)
    prompt_key: str          # key into experiment_prompts.PROMPTS
    main_metric: str         # metric plotted / range-checked for this group
    cases: tuple
    relations: tuple = ()
    expectations: dict = field(default_factory=dict)


# ---------------------------------------------------------------- A-1 speed

A1 = Group(
    'A1', '부사에 따른 선속도', 'A', 'single', 'A', 'abs_linear_x',
    cases=(
        Case('A1-01', '앞으로 이동해', linear_sign=1, angular_sign=0, tags=('baseline',)),
        Case('A1-02', '앞으로 아주 천천히 이동해', linear_sign=1, angular_sign=0),
        Case('A1-03', '앞으로 천천히 이동해', linear_sign=1, angular_sign=0),
        Case('A1-04', '앞으로 살살 이동해', linear_sign=1, angular_sign=0),
        Case('A1-05', '앞으로 느긋하게 이동해', linear_sign=1, angular_sign=0),
        Case('A1-06', '앞으로 조금 빠르게 이동해', linear_sign=1, angular_sign=0),
        Case('A1-07', '앞으로 빠르게 이동해', linear_sign=1, angular_sign=0),
        Case('A1-08', '앞으로 아주 빠르게 이동해', linear_sign=1, angular_sign=0),
        Case('A1-09', '앞으로 최대한 빠르게 이동해', linear_sign=1, angular_sign=0),
        Case('A1-10', '앞으로 조심스럽게 이동해', linear_sign=1, angular_sign=0),
    ),
    relations=(
        # intensity chain
        Relation('abs_linear_x', 'A1-02', '<', 'A1-03'),
        Relation('abs_linear_x', 'A1-03', '<', 'A1-06'),
        Relation('abs_linear_x', 'A1-06', '<', 'A1-07'),
        Relation('abs_linear_x', 'A1-07', '<', 'A1-08'),
        Relation('abs_linear_x', 'A1-08', '<=', 'A1-09'),
        # neutral command sits between slow and fast (ties allowed)
        Relation('abs_linear_x', 'A1-03', '<=', 'A1-01'),
        Relation('abs_linear_x', 'A1-01', '<=', 'A1-07'),
        # every slow-sounding adverb is slower than "조금 빠르게"
        Relation('abs_linear_x', 'A1-04', '<', 'A1-06'),
        Relation('abs_linear_x', 'A1-05', '<', 'A1-06'),
        Relation('abs_linear_x', 'A1-10', '<', 'A1-06'),
    ),
)

# -------------------------------------------------------------- A-2 yaw rate

A2 = Group(
    'A2', '부사에 따른 각속도', 'A', 'single', 'A', 'abs_angular_z',
    cases=(
        Case('A2-01', '오른쪽으로 회전해', linear_sign=0, angular_sign=-1, tags=('baseline',)),
        Case('A2-02', '오른쪽으로 아주 천천히 회전해', linear_sign=0, angular_sign=-1),
        Case('A2-03', '오른쪽으로 천천히 회전해', linear_sign=0, angular_sign=-1),
        Case('A2-04', '오른쪽으로 살살 회전해', linear_sign=0, angular_sign=-1),
        Case('A2-05', '오른쪽으로 조금 빠르게 회전해', linear_sign=0, angular_sign=-1),
        Case('A2-06', '오른쪽으로 빠르게 회전해', linear_sign=0, angular_sign=-1),
        Case('A2-07', '오른쪽으로 아주 빠르게 회전해', linear_sign=0, angular_sign=-1),
        Case('A2-08', '오른쪽으로 최대한 빠르게 회전해', linear_sign=0, angular_sign=-1),
        Case('A2-09', '왼쪽으로 천천히 회전해', linear_sign=0, angular_sign=1),
        Case('A2-10', '왼쪽으로 빠르게 회전해', linear_sign=0, angular_sign=1),
    ),
    relations=(
        Relation('abs_angular_z', 'A2-02', '<', 'A2-03'),
        Relation('abs_angular_z', 'A2-03', '<', 'A2-05'),
        Relation('abs_angular_z', 'A2-05', '<', 'A2-06'),
        Relation('abs_angular_z', 'A2-06', '<', 'A2-07'),
        Relation('abs_angular_z', 'A2-07', '<=', 'A2-08'),
        Relation('abs_angular_z', 'A2-03', '<=', 'A2-01'),
        Relation('abs_angular_z', 'A2-01', '<=', 'A2-06'),
        Relation('abs_angular_z', 'A2-04', '<', 'A2-05'),
        Relation('abs_angular_z', 'A2-09', '<', 'A2-10'),
    ),
    # Same adverb, opposite direction: magnitudes should match.
    expectations={'mirror_pairs': (('A2-03', 'A2-09'), ('A2-06', 'A2-10'))},
)

# ------------------------------------------------------------ B distance

# Reference intervals are rough human-scale lengths (adult body measures and
# common furniture). They are evaluation aids only; ambiguous wording has none.
B = Group(
    'B', '자연어 기반 이동거리', 'B', 'single', 'B', 'distance',
    cases=(
        Case('B-01', '앞으로 한 뼘 이동해', linear_sign=1, angular_sign=0,
             reference_range=(0.15, 0.25), note='한 뼘 ≈ 엄지~새끼손가락 끝, 약 0.2 m'),
        Case('B-02', '앞으로 두 뼘 이동해', linear_sign=1, angular_sign=0,
             reference_range=(0.30, 0.50), note='한 뼘의 2배'),
        Case('B-03', '앞으로 한 발짝 이동해', linear_sign=1, angular_sign=0,
             reference_range=(0.30, 0.80), note='발 길이(~0.27 m)~보폭(~0.7 m) 사이로 해석 가능'),
        Case('B-04', '앞으로 두 걸음 이동해', linear_sign=1, angular_sign=0,
             reference_range=(0.90, 1.60), note='성인 보폭 0.6~0.8 m의 2배'),
        Case('B-05', '앞으로 팔 하나 길이만큼 이동해', linear_sign=1, angular_sign=0,
             reference_range=(0.55, 0.85), note='어깨~손끝 약 0.7 m'),
        Case('B-06', '앞으로 손바닥 길이만큼 이동해', linear_sign=1, angular_sign=0,
             reference_range=(0.08, 0.22), note='손바닥만(~0.1 m) 또는 손 전체(~0.19 m)'),
        Case('B-07', '앞으로 아주 조금 이동해', linear_sign=1, angular_sign=0,
             note='기준 거리 없음(모호), 순서만 평가', tags=('ambiguous',)),
        Case('B-08', '앞으로 조금 이동해', linear_sign=1, angular_sign=0,
             note='기준 거리 없음(모호), 순서만 평가', tags=('ambiguous',)),
        Case('B-09', '앞으로 꽤 멀리 이동해', linear_sign=1, angular_sign=0,
             note='기준 거리 없음(모호), 순서만 평가', tags=('ambiguous',)),
        Case('B-10', '앞으로 책상 길이만큼 이동해', linear_sign=1, angular_sign=0,
             reference_range=(1.00, 1.80), note='일반 책상 폭 1.0~1.8 m, 책상 종류에 따라 다름'),
    ),
    relations=(
        Relation('distance', 'B-01', '<', 'B-02'),
        Relation('distance', 'B-03', '<', 'B-04'),
        Relation('distance', 'B-06', '<=', 'B-01'),
        Relation('distance', 'B-01', '<', 'B-05'),
        Relation('distance', 'B-07', '<', 'B-08'),
        Relation('distance', 'B-08', '<', 'B-09'),
        Relation('distance', 'B-06', '<', 'B-10'),
    ),
    # "두 X" should be about twice "한 X".
    expectations={'ratio_pairs': (('B-01', 'B-02', 2.0), ('B-03', 'B-04', 2.0))},
)

# ------------------------------------------------------------ C duration

C = Group(
    'C', '시간 관련 자연어 표현', 'C', 'single', 'C', 'duration',
    cases=(
        Case('C-01', '앞으로 이동해', linear_sign=1, angular_sign=0, tags=('baseline',)),
        Case('C-02', '앞으로 아주 잠깐 이동해', linear_sign=1, angular_sign=0),
        Case('C-03', '앞으로 잠깐 이동해', linear_sign=1, angular_sign=0),
        Case('C-04', '앞으로 잠시 이동해', linear_sign=1, angular_sign=0),
        Case('C-05', '앞으로 잠시만 이동해', linear_sign=1, angular_sign=0),
        Case('C-06', '앞으로 조금만 이동해', linear_sign=1, angular_sign=0,
             note='시간이 아니라 거리(이동량)로도 해석될 수 있음', tags=('distance_ambiguous',)),
        Case('C-07', '앞으로 한동안 이동해', linear_sign=1, angular_sign=0),
        Case('C-08', '앞으로 오래 이동해', linear_sign=1, angular_sign=0),
        Case('C-09', '앞으로 계속 이동해', note='종료 시간이 정해지지 않은 명령',
             tags=('indefinite',)),
        Case('C-10', '앞으로 멈추라고 할 때까지 이동해', note='종료 시간이 정해지지 않은 명령',
             tags=('indefinite',)),
    ),
    relations=(
        Relation('duration', 'C-02', '<=', 'C-03'),
        Relation('duration', 'C-02', '<', 'C-07'),
        Relation('duration', 'C-03', '<', 'C-07'),
        Relation('duration', 'C-04', '<', 'C-07'),
        Relation('duration', 'C-05', '<', 'C-07'),
        Relation('duration', 'C-07', '<=', 'C-08'),
        Relation('duration', 'C-03', '<', 'C-08'),
        # 잠깐 / 잠시 / 잠시만 are near-synonyms: deliberately no order between them.
    ),
)

# --------------------------------------------------------- D geometric paths

_D_CASES = (
    Case('D-01', '제자리에서 유턴해', linear_sign=0, tags=('uturn', 'in_place')),
    Case('D-02', '앞으로 이동하면서 유턴해', linear_sign=1, tags=('uturn', 'moving')),
    Case('D-03', '오른쪽으로 원을 그리면서 이동해', linear_sign=1, angular_sign=-1,
         tags=('circle',)),
    Case('D-04', '왼쪽으로 원을 그리면서 이동해', linear_sign=1, angular_sign=1,
         tags=('circle',)),
    Case('D-05', '작은 원을 그리면서 이동해', linear_sign=1, tags=('circle', 'small')),
    Case('D-06', '큰 원을 그리면서 이동해', linear_sign=1, tags=('circle', 'large')),
    Case('D-07', '포물선을 그리면서 이동해', tags=('parabola',),
         note='일정한 Twist 하나로는 포물선을 정확히 표현할 수 없음'),
    Case('D-08', '직사각형을 그리면서 이동해', tags=('rectangle',),
         note='직진과 90° 회전의 반복이 필요'),
    Case('D-09', '정사각형을 그리면서 이동해', tags=('square',),
         note='네 변의 길이가 같아야 함'),
    Case('D-10', '지그재그로 이동해', tags=('zigzag',),
         note='좌우 회전 방향이 번갈아 바뀌어야 함'),
)

D1 = Group(
    'D1', '복합 동작 (단일 Twist)', 'D', 'single', 'D1', 'rotation_deg',
    cases=_D_CASES,
    relations=(Relation('radius', 'D-05', '<', 'D-06'),),
    expectations={
        'uturn_deg': 180.0,
        # A single constant Twist can only make a straight line, an in-place
        # turn or a circular arc.
        'single_twist_representable': ('D-01', 'D-02', 'D-03', 'D-04', 'D-05', 'D-06'),
    },
)

D2 = Group(
    'D2', '복합 동작 (동작 시퀀스)', 'D', 'actions', 'D2', 'rotation_deg',
    cases=_D_CASES,
    relations=(Relation('radius', 'D-05', '<', 'D-06'),),
    expectations={'uturn_deg': 180.0, 'corner_deg': 90.0},
)

GROUPS = (A1, A2, B, C, D1, D2)
EXPERIMENTS = {
    'A': (A1, A2),
    'B': (B,),
    'C': (C,),
    'D': (D1, D2),
}


def groups_for(experiment):
    """'all' or one of A/B/C/D -> tuple of groups."""
    if experiment == 'all':
        return GROUPS
    try:
        return EXPERIMENTS[experiment]
    except KeyError:
        raise ValueError(f'unknown experiment {experiment!r}; use A, B, C, D or all') from None


def group_by_id(group_id):
    for group in GROUPS:
        if group.group_id == group_id:
            return group
    raise KeyError(group_id)
