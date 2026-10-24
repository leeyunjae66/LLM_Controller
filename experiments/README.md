# LLM 자연어 운동 추론 평가 실험

로컬 Ollama의 `qwen3:4b`가 정성적인 자연어 표현(부사, 일상적인 거리, 시간 표현, 도형)을
로봇 운동 파라미터(`linear_x`, `angular_z`, `duration`)로 얼마나 합리적이고 일관되게
바꾸는지 평가합니다.

- **LLM 출력만 분석합니다.** ROS 노드를 띄우지 않고, `/cmd_vel`을 포함한 어떤 토픽에도 발행하지 않습니다.
- 패키지의 `OllamaClient`(통신·Timeout·오류 처리)와 `validator.py`(검증 규칙)를 그대로 재사용합니다.
- 기존 `prompt.py`, `validator.py`, `velocity_node.py`는 수정하지 않습니다. 실험용 프롬프트는 따로 관리합니다.

## 파일

| 파일 | 역할 |
|---|---|
| `run_llm_experiment.py` | 실험 실행 (CLI) |
| `experiment_prompts.py` | 실험별 System Prompt와 JSON Schema (단일 Twist / 동작 시퀀스) |
| `experiment_cases.py` | 자연어 명령 60개와 **평가 기준**(의미 순서 관계, 기대 부호, 참고 거리 범위). LLM에는 보내지 않음 |
| `experiment_core.py` | 응답 파싱·검증, 거리·회전각·반지름 계산, 이상적 XY 궤적 적분 |
| `experiment_clients.py` | `ExperimentOllamaClient`(패키지 `OllamaClient` 상속), 서버·모델 사전 점검, 오프라인용 Mock |
| `analyze_results.py` | 통계, 의미 순서 일치율, 그래프, 보고서 생성 |
| `results/` | 실행 결과 저장 위치 (`results/<실행 ID>/`, git에는 올라가지 않음) |

## 실험 구성

| 실험 | 그룹 | 모드 | 주 지표 | 내용 |
|---|---|---|---|---|
| A | A1 | 단일 | \|linear_x\| | 부사에 따른 선속도 (아주 천천히 … 최대한 빠르게) |
| A | A2 | 단일 | \|angular_z\| + 부호 | 부사에 따른 각속도, 좌우 방향 |
| B | B | 단일 | `linear_x × duration` | 한 뼘, 한 발짝, 책상 길이 등 거리 표현 |
| C | C | 단일 | duration | 잠깐, 한동안, 계속, 멈추라고 할 때까지 등 시간 표현 |
| D | D1 | 단일 | 회전각, 반지름 | 유턴·원·도형을 **Twist 하나**로 표현할 수 있는지 |
| D | D2 | 시퀀스 | 궤적 | 같은 명령을 `{"actions": [...]}` **동작 시퀀스**로 |

D는 같은 10개 명령을 두 가지 출력 구조로 각각 실행합니다. 단일 모드에서 직사각형처럼 표현할 수 없는
명령은 LLM이 정지 형식(모두 0)으로 답해야 정직한 답입니다.

## 프롬프트 설계 원칙

`experiment_prompts.py`의 프롬프트는 다음을 지킵니다(`test/test_experiments.py`에서 자동 검사).

- 출력 필드의 물리적 의미와 부호(앞 +, 왼쪽/반시계 +)를 설명하고, `거리 = linear_x × duration`,
  `회전각 = angular_z × duration`, `반지름 = |linear_x / angular_z|` 관계만 알려 줍니다.
- **예시 값·기본값·단어-숫자 매핑이 없습니다.** 프롬프트에 나오는 숫자는 로봇 한계값, 정지 형식의 0,
  최대 동작 수뿐입니다. 테스트 명령, 기대 순서, 참고 거리 범위도 들어가지 않습니다.
- 명시된 숫자와 단위는 그대로 보존하고, 한계를 넘어도 고치지 말라고 지시합니다. 범위는 JSON Schema에도
  넣지 않아서, 범위를 벗어난 선택도 그대로 기록되고 Validator가 판정합니다.
- 표현할 수 없거나 불명확한 명령은 꾸며내지 말고 정지 형식(`0,0,0` 또는 `{"actions": []}`)으로 답하게 합니다.
  "계속 이동해" 같은 무기한 명령을 어떻게 처리할지는 **지시하지 않습니다**(그 자체가 관찰 대상).
- A·B·C·D1은 일부러 같은 프롬프트를 씁니다. 실험 간 차이가 프롬프트 문구가 아니라 명령에서 오게 하려는
  것이고, 실험마다 따로 바꿀 수 있도록 항목은 나눠 두었습니다.
- `--prompt-variant no_limits`로 로봇 한계를 알려 주지 않은 조건도 비교할 수 있습니다.

로봇 한계값은 실행할 때 `validator.py`에서 읽습니다. 작업 트리에서 값을 바꿨다면 그 값이 쓰이고,
`metadata.json`과 `report.md`에 사용한 한계값과 커밋되지 않은 변경 여부가 기록됩니다.

## 실행

ROS를 source하거나 colcon build할 필요가 없습니다. Ollama만 켜 두면 됩니다.

```bash
ollama serve                       # 꺼져 있다면 다른 터미널에서
cd ~/ros2_ws/src/llm_velocity_controller/experiments
python3 run_llm_experiment.py --experiment all --repeat 10
python3 run_llm_experiment.py --experiment A --repeat 10      # B, C, D도 같은 방식
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--experiment` | `all` | `A`, `B`, `C`, `D`, `all` |
| `--repeat` | `10` | 명령당 반복 횟수 |
| `--model` | `qwen3:4b` | Ollama 모델 |
| `--api-url` | `http://localhost:11434/api/chat` | Ollama 주소 |
| `--temperature` | `0.0` | 샘플링 온도 |
| `--seed` | 없음 | Ollama 시드 (temperature > 0일 때 재현용) |
| `--timeout` | `60` | 요청당 Timeout(초) |
| `--prompt-variant` | `with_limits` | `no_limits`면 로봇 한계를 알려 주지 않음 |
| `--cases` | 전체 | 일부만 실행, 예: `--cases A1-02,B-04,D-08` |
| `--limit-cases` | 0(전체) | 그룹마다 앞에서 N개만 (빠른 점검용) |
| `--backend` | `ollama` | `mock`: Ollama 없이 파이프라인만 확인 (**결과는 LLM 결과가 아님**) |
| `--output-dir`, `--run-id` | `results/`, 타임스탬프 | 저장 위치 |
| `--no-analyze` | – | 추론만 하고 분석은 나중에 |

- 시작하기 전에 서버 연결과 모델 설치 여부를 확인하고, 문제가 있으면 원인과 해결 명령을 출력한 뒤 종료합니다
  (`ollama serve` 실행, `ollama pull qwen3:4b`).
- 개별 요청의 Timeout·연결 오류·잘못된 응답은 기록만 하고 실험을 계속합니다.
- 중간에 Ctrl+C를 누르면 그때까지의 결과를 저장하고 분석합니다. `raw.csv`는 한 줄씩 바로 기록됩니다.
- 소요 시간: 이 PC(GPU 없음)에서는 응답 하나에 수 초 걸립니다. `all --repeat 10`은 600회 추론이라
  한 시간 정도 걸릴 수 있습니다.

분석만 다시 하려면:

```bash
python3 analyze_results.py results/<실행 ID>
```

## 결과 파일 (`results/<실행 ID>/`)

| 파일 | 내용 |
|---|---|
| `raw.csv` | 추론 1회당 1행 (아래 필드) |
| `raw.json` | 위 내용 + 동작 시퀀스(`actions`), D 실험의 궤적(`path`: `[t, x, y, θ]`), Ollama 서버 측 시간 |
| `metadata.json` | 모델, 옵션, 반복 수, 한계값, **사용한 프롬프트 원문과 Schema**, git 커밋·작업 트리 상태 |
| `summary.csv` / `summary.json` | 명령별 통계와 그룹별 지표 |
| `report.md` | 사람이 읽는 요약 보고서 |
| `plots/*.png` | 그래프 7개 |

`raw.csv` 필드: `experiment, group, mode, case_id, command, repeat, linear_x, angular_z, duration,
estimated_distance, estimated_rotation_rad, estimated_rotation_deg, radius, n_actions, path_length,
closure_distance, inference_time_sec, json_valid, validation_passed, validation_error, refused,
limit_violations, error_type, error, raw_response`

- `estimated_distance = linear_x × duration`: 등속을 가정한 이상적인 거리이며 실제 주행거리가 아닙니다.
  시퀀스 모드에서는 구간별 합(부호 포함), `path_length`는 이동한 길이의 합입니다.
- `json_valid`: 세 필드가 모두 있는 숫자 JSON인가. `validation_passed`: `validator.py` 범위 검사까지 통과했는가.
- `refused`: 정지 형식(속도가 모두 0)으로 답했는가.
- `limit_violations`: 어떤 한계를 넘었는가(`linear_x`, `angular_z`, `duration`).

## 지표와 해석

**값 통계**: 명령마다 평균, 표준편차, 최소, 최대를 냅니다. JSON이 유효한 응답은 **Validator가 거부한 것도 포함**합니다.
"최대한 빠르게"에 한계를 넘는 값을 고른 것도 모델의 판단이기 때문입니다. 통과율은 따로 봅니다.

| 지표 | 의미 |
|---|---|
| JSON 유효율 / Validator 통과율 | 형식을 지켰는지 / 실행 가능한 값이었는지 |
| 동일 출력 비율 | 반복 중 가장 많이 나온 출력의 비율. temperature=0이면 100%에 가까운 게 정상 |
| 평균 추론 시간 | 요청부터 응답까지의 벽시계 시간 |
| **의미 순서 일치율** | `experiment_cases.py`의 관계 중 평균이 순서를 지킨 비율 |
| 부호 정확도 | 앞/뒤, 왼쪽/오른쪽 방향이 맞은 비율 |
| 기준 범위 내 비율 (B) | 참고 거리 구간에 들어간 비율 |
| 종료 처리 분류 (C) | `finite`(유한 시간 선택), `refused`(정지), `over_limit`(한계 초과로 거부), `invalid`, `error` |
| 형상 검사 (D) | 유턴 180±20°, 제자리 여부, 원 한 바퀴(≥330°), 사각형의 직각(90±15°)·폐곡선·변 길이, 지그재그 회전 방향 교대 |

**의미 순서 관계**는 문장 의미로 순서가 분명한 쌍만 정의했습니다(예: 아주 천천히 < 천천히 < 조금 빠르게 < 빠르게).
비교할 수 없는 표현(살살 vs 느긋하게, 잠깐 vs 잠시 vs 잠시만)에는 순위를 매기지 않습니다. `<`는 엄격한
순서라서 평균이 같으면 "동률"로 따로 표시하고 일치로 세지 않습니다. `<=`는 같아도 일치입니다.

**참고 거리 범위(B)** 는 사람의 신체 치수와 일반 가구 크기를 기준으로 잡은 대략적인 구간이고 정답이 아닙니다.
"아주 조금", "조금", "꽤 멀리"처럼 기준 거리가 없는 표현은 범위 없이 순서만 평가합니다. 사람이 직접 평가할
때는 `raw.csv`의 `estimated_distance`와 `report.md`의 메모를 함께 보세요. 이동거리의 상한은
`linear_x 한계 × duration 한계`라서, 한계값에 따라 "책상 길이"처럼 긴 거리는 표현 자체가 불가능할 수 있습니다.

**해석할 때 주의할 점**

- temperature=0에서는 반복해도 거의 같은 답이 나옵니다. 이때 표준편차가 0인 것은 "안정적"이라는 뜻이지
  "정확하다"는 뜻이 아닙니다. 다양성을 보려면 `--temperature 0.7 --seed …`처럼 바꿔 비교하세요.
- 평가 기준(순서 관계, 범위, 형상 허용 오차)은 이 저장소에서 정한 것입니다. 보고서의 수치는 그 기준에 대한 결과입니다.
- `--backend mock` 결과는 규칙 기반 가짜 값입니다. 보고서 맨 위에 경고가 붙으며, LLM 성능으로 해석하면 안 됩니다.

## 그래프 (`plots/`)

| 파일 | 내용 |
|---|---|
| `a1_linear_speed.png` | 부사별 평균 \|linear_x\| ± 표준편차, 한계선 |
| `a2_angular_speed.png` | 부사별 평균 angular_z ± 표준편차 (빨강 = 오른쪽/음수) |
| `b_distance.png` | 거리 표현별 추정 이동거리, 초록 점선 = 참고 범위 |
| `c_duration.png` | 시간 표현별 duration, 한계선 |
| `repeat_distribution.png` | 같은 명령을 반복했을 때 출력 분포 (A1, A2, B, C) |
| `d1_paths_single.png` / `d2_paths_actions.png` | 복합 동작의 이상적 XY 궤적 (원점에서 +x를 보고 출발) |

한글 폰트(Noto CJK, 나눔 등)가 있으면 축에 명령 문장을, 없으면 case ID만 표시합니다. 폰트가 없어도 실행은 멈추지 않습니다.
pandas는 쓰지 않습니다(표준 라이브러리로 통계 계산). matplotlib이 없으면 그래프만 건너뜁니다.

## 테스트

Ollama 없이 실행됩니다.

```bash
cd ~/ros2_ws/src/llm_velocity_controller
python3 -m pytest -q test/test_experiments.py
```

케이스·관계 무결성, 프롬프트에 테스트 문장·부사·예시 수치가 새지 않는지, 응답 파싱과 검증, 궤적 적분,
순서 일치·동률 판정, mock 전체 실행과 파일 생성, 요청 오류가 나도 실험이 계속되는지, 서버 연결 실패와
모델 미설치 메시지를 검사합니다.
