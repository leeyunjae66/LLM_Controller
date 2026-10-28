# llm_velocity_controller

자연어 이동 명령을 로컬 LLM(Ollama + `qwen3:4b`)이 JSON 속도 명령으로 바꾸고,
Validator가 검증한 뒤 `geometry_msgs/msg/Twist`로 **`/cmd_vel`**에
발행하는 ROS 2 Humble 패키지입니다. 로봇이 다른 토픽을 구독하면 remap합니다(예: RUBI는 `/rubi_gazebo/cmd_vel`).

```text
/llm/user_command (std_msgs/String)
   │
   ├─ STOP 키워드(정지·멈춰·그만·비상·stop…) ──► 즉시 0속도 (LLM 호출 없음)
   │
   └─ Worker Thread: prompt.py → llm_client.py (Ollama /api/chat) → validator.py
                                          │ (generation, 결과|오류) 큐
                                          ▼
                     10 Hz ROS Timer ──► /cmd_vel (geometry_msgs/Twist)
```

## 모듈

| 파일 | 역할 |
|---|---|
| `prompt.py` | System Prompt, 메시지 구성, Structured Output용 JSON Schema, 로컬 STOP 판별(`is_stop_command`) |
| `llm_client.py` | `OllamaClient`(표준 라이브러리 `urllib`), 연결/Timeout/응답 오류 예외, 원본 응답 로깅, warm-up. `MockLLMClient`(Ollama 없이 실행용) |
| `validator.py` | ROS 비의존 순수 Python 검증기. 범위 밖 값은 **보정하지 않고 거부** |
| `velocity_node.py` | `llm_velocity_controller` 노드. Worker Thread + 10 Hz Timer, generation 카운터, 상태 토픽 `/llm/status` |
| `launch/llm_velocity_controller.launch.py` | Ollama(필요할 때)·velocity_node·chat 창을 한 번에 실행 |
| `chat.py` | 터미널 채팅 입력기(`ros2 run llm_velocity_controller chat`). 입력한 문장을 `/llm/user_command`로 보내고 `/llm/status`를 같은 화면에 표시 |

### JSON 계약과 허용 범위

```json
{"linear_x": 0.2, "angular_z": 0.0, "duration": 1.0}
```

| 필드 | 범위 | 비고 |
|---|---|---|
| `linear_x` | -1.0 ~ 1.0 m/s | 앞(+), 뒤(-) |
| `angular_z` | -0.5 ~ 0.5 rad/s | 왼쪽(+, 반시계), 오른쪽(-) |
| `duration` | 0 ~ 10.0 s | 음수 거부 |

거부 조건: JSON 파싱 실패(코드블록·꼬리 문자열 포함), 객체가 아님, 필드 누락, 추가 필드,
중복 키, 숫자가 아님(`true/false`, 문자열, `null`), `NaN`/`Infinity`/오버플로, 범위 초과.
범위는 `validator.py`의 `LINEAR_X_LIMIT`, `ANGULAR_Z_LIMIT`, `DURATION_MAX`이고, System Prompt에도 이 값이 그대로 들어갑니다.

System Prompt는 단어별 고정값(예: "천천히 = 0.1")을 주지 않고, 명시되지 않은 속도와 시간을 LLM이 명령의 의도에 맞게
위 범위 안에서 직접 고르게 합니다. 사용자가 숫자를 명시하면 범위를 넘어도 그대로 두고(Validator가 거부),
불명확하거나 지원하지 않는 명령은 정지(모두 0)로, "계속 가" 같은 무기한 명령은 유한한 시간이나 정지로 답하게 합니다.
`prompt.py`의 `DEFAULT_*` 값(0.1 m/s, 0.3 rad/s, 5초)은 Ollama 없이 쓰는 `MockLLMClient`에서만 씁니다.

### 노드 동작

- 새 명령을 받으면 **즉시** 이전 동작을 취소하고 0속도를 발행한 뒤 LLM에 질의합니다.
- STOP 키워드는 LLM을 기다리지 않고 그 자리에서 정지합니다. 키워드가 들어간 문장은
  모두 정지로 처리합니다(예: "정지하지 말고 가"도 정지). 안전 쪽으로 치우친 의도된 동작입니다.
- LLM 호출은 요청마다 별도 daemon 스레드에서 실행되고, 결과는 `(generation, 결과, 오류)`로 큐에만 넣습니다.
  Publisher는 executor 스레드(구독 콜백, Timer)에서만 사용합니다.
- 10 Hz Timer가 큐를 비우며, generation이 최신이 아닌 결과는 버립니다(늦게 온 응답이 새 명령을 덮어쓰지 않음).
- 실행 시간은 `time.monotonic()` 기준. 지속시간이 끝나면 0속도를 한 번 발행합니다.
- LLM 오류(연결 실패·Timeout·잘못된 응답)와 검증 실패 시 0속도를 발행합니다.
- 종료(Ctrl+C, SIGINT, SIGTERM) 시 진행 중인 요청을 무효화하고 0속도를 발행합니다.
- Twist의 `linear.x`, `angular.z` 외 필드는 항상 0입니다.
- 처리 과정을 `/llm/status`(`std_msgs/String`, JSON 한 줄)로 알립니다.

### 상태 토픽 `/llm/status`

`data`는 JSON 객체 하나이고, `id`는 명령 번호(generation)입니다.

| `state` | 의미 | 추가 필드 |
|---|---|---|
| `querying` | 명령을 받아 LLM에 질의 중 | `text` |
| `stopped` | STOP 키워드, LLM 없이 즉시 정지 | `text` |
| `executing` | 검증 통과, 속도 발행 시작 | `linear_x`, `angular_z`, `duration` |
| `done` | 지속시간 종료, 0속도 발행 | |
| `llm_stop` | LLM이 정지(모두 0)로 해석 | |
| `rejected` | LLM 오류 또는 검증 실패, 0속도 발행 | `detail` |
| `stale` | 이전 명령의 늦은 응답이라 무시 | |
| `shutdown` | 노드 종료, 0속도 발행 | |

```json
{"id": 1, "state": "executing", "linear_x": 0.2, "angular_z": 0.0, "duration": 1.0}
```

### 파라미터

| 이름 | 기본값 | 설명 |
|---|---|---|
| `llm_backend` | `ollama` | `ollama` 또는 `mock`(Ollama 없이 규칙 기반 응답) |
| `model` | `qwen3:4b` | Ollama 모델 이름 |
| `api_url` | `http://localhost:11434/api/chat` | Ollama chat 엔드포인트 |
| `timeout_sec` | `30.0` | LLM 요청 Timeout |
| `keep_alive` | `30m` | 모델을 메모리에 유지하는 시간 |
| `warm_up` | `true` | 노드 시작 시 모델을 미리 로드 |
| `log_raw_response` | `true` | LLM 원본 응답을 INFO 로그로 출력 |
| `mock_delay_sec` | `0.0` | mock 응답 지연(느린 LLM 흉내) |

## 설치

```bash
# Ollama와 모델 (이미 설치되어 있으면 생략)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# 서버가 꺼져 있으면
ollama serve
curl http://localhost:11434/api/version
```

Python 의존성은 표준 라이브러리와 ROS 2 기본 패키지(`rclpy`, `std_msgs`, `geometry_msgs`)뿐이고,
테스트에는 `python3-pytest`가 필요합니다.

## 빌드

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_velocity_controller
source install/setup.bash
```

## 실행

### 한 번에 실행하기 (launch)

```bash
ros2 launch llm_velocity_controller llm_velocity_controller.launch.py
```

한 명령으로 다음을 모두 띄웁니다.

1. **Ollama**: `api_url`에서 응답이 없으면 `ollama serve`를 직접 시작하고, 3초 뒤 노드를 띄웁니다.
   이미 실행 중이면 그대로 씁니다. launch가 시작한 Ollama는 launch를 끌 때 같이 꺼집니다.
2. **velocity_node**: launch를 띄운 터미널에 로그가 나옵니다.
3. **chat**: **새 터미널 창**(gnome-terminal)이 열리고 그 창에서 명령을 입력합니다.
   `ros2 launch`는 키보드 입력을 프로세스에 전달하지 않기 때문에 창을 따로 띄웁니다.

종료하는 방법은 두 가지이고, 어느 쪽이든 0속도를 발행한 뒤 전부 꺼집니다.
- chat 창에서 `exit`를 입력하거나 창을 닫기
- launch 터미널에서 Ctrl+C (chat 창도 스스로 닫힙니다)

| launch 인자 | 기본값 | 설명 |
|---|---|---|
| `llm_backend` | `ollama` | `mock`이면 Ollama 없이 실행 |
| `model` | `qwen3:4b` | Ollama 모델 |
| `api_url` | `http://localhost:11434/api/chat` | Ollama 주소 |
| `timeout_sec` | `30.0` | LLM 요청 Timeout |
| `keep_alive` | `30m` | 모델 유지 시간 |
| `log_raw_response` | `true` | LLM 원본 응답 로그 |
| `start_ollama` | `auto` | `auto`: 응답이 없을 때만 `ollama serve` 시작, `false`: 시작하지 않음 |
| `chat` | `true` | chat 창 열기 |
| `chat_terminal` | `gnome-terminal --wait --title=LLM-chat --` | chat 창을 여는 명령. 예: `'terminator -x'` |
| `shutdown_with_chat` | `true` | chat 창이 닫히면 전체 종료 |
| `cmd_vel_topic` | `/cmd_vel` | 속도 출력 토픽 remap (RUBI: `/rubi_gazebo/cmd_vel`) |

```bash
# Ollama 없이 동작만 확인
ros2 launch llm_velocity_controller llm_velocity_controller.launch.py llm_backend:=mock
# 노드만 띄우고 chat은 직접 실행 (SSH처럼 창을 띄울 수 없는 환경)
ros2 launch llm_velocity_controller llm_velocity_controller.launch.py chat:=false
# RUBI Gazebo 구동
ros2 launch llm_velocity_controller llm_velocity_controller.launch.py cmd_vel_topic:=/rubi_gazebo/cmd_vel
```

### 따로 실행하기

```bash
ros2 run llm_velocity_controller velocity_node
# Ollama 없이:
ros2 run llm_velocity_controller velocity_node --ros-args -p llm_backend:=mock
```

### 터미널 채팅으로 명령하기 (권장)

launch를 쓰지 않을 때는 노드를 실행한 뒤 다른 터미널에서:

```bash
ros2 run llm_velocity_controller chat
```

```text
> 앞으로 1초 동안 0.2m/s로 이동해
  [#1] LLM 해석 중...
  [#1] 실행: 직진 +0.20 m/s, 회전 +0.00 rad/s, 1.0초 동안
  [#1] 완료 (0속도 발행)
> 앞으로 10m/s로 가
  [#2] LLM 해석 중...
  [#2] 거부됨, 0속도 발행: ValidationError: linear_x 10.0 outside [-1.0, 1.0] m/s
> 정지
  [#3] 정지 (LLM 없이 즉시 처리)
```

- 상태 메시지가 도착해도 입력 중인 글자는 지워지지 않고 프롬프트 아래에 다시 그려집니다.
  방향키로 줄 편집과 이전 입력 불러오기가 됩니다(readline).
- 빈 줄은 보내지 않습니다. `exit`, `quit`, `q`, `종료`, Ctrl+C, Ctrl+D로 끝내며, **끝낼 때 정지 명령을 보냅니다.**
- LLM이 응답하는 동안에도 `정지`를 바로 입력할 수 있습니다.
- velocity_node가 보이지 않으면 시작할 때 경고를 출력합니다.
- 파이프 입력도 됩니다: `printf '앞으로 이동해\n' | ros2 run llm_velocity_controller chat`
  (입력이 끝나면 곧바로 정지를 보내고 종료하므로, 결과를 보려면 입력 뒤에 대기를 넣으세요.)

### 토픽으로 직접 명령하기

다른 터미널에서 명령 발행과 결과 확인:

```bash
ros2 topic pub --once /llm/user_command std_msgs/msg/String "{data: '앞으로 1초 동안 0.2m/s로 이동해'}"
ros2 topic echo /cmd_vel
```

`linear.x: 0.2`가 약 1초(10 Hz로 10개) 나온 뒤 0이 한 번 발행됩니다.
CPU에서만 추론하는 환경에서는 LLM 응답까지 약 4~5초가 걸리므로, 그만큼 늦게 움직이기 시작합니다.

> 같은 네트워크의 다른 PC와 토픽이 섞이지 않게 하려면 모든 터미널에서
> `export ROS_LOCALHOST_ONLY=1`(또는 고유한 `ROS_DOMAIN_ID`)을 설정하세요.

## 테스트

```bash
cd ~/ros2_ws
colcon test --packages-select llm_velocity_controller
colcon test-result --verbose --test-result-base build/llm_velocity_controller

# 또는 직접
source install/setup.bash
cd src/llm_velocity_controller && python3 -m pytest -q -p no:cacheprovider test
```

| 파일 | 내용 |
|---|---|
| `test/test_validator.py` | 정상 값·경계값, 파싱 실패, 비객체, 필드 누락/추가/중복, 타입, NaN/Infinity/오버플로, 범위 초과(보정 없음), 음수 duration |
| `test/test_prompt_and_client.py` | STOP 판별, 프롬프트/스키마 계약, 요청 본문(`stream=false`, `think=false`, `format`, `keep_alive`), Timeout·연결 실패·HTTP 오류·잘못된 응답 본문, warm-up, Mock 클라이언트 |
| `test/test_velocity_node.py` | 실제 LLM 없이 가짜 LLM·가짜 시계·발행 기록기로 노드 검증: duration 후 0, 즉시 STOP, 느린 추론 중 STOP, 명령 취소, 오래된 응답 폐기, Timeout/잘못된 JSON/범위 초과 시 0, 종료 시 0, 10 Hz 주기, 출력 토픽 고정, 상태 이벤트 순서 |
| `test/test_launch.py` | launch 인자 선언, Ollama 재사용·자동 시작·`start_ollama:=false`, chat 창 prefix와 종료 연동, mock 백엔드 구성 |
| `test/test_chat.py` | 상태 파싱·문구, 종료 단어, 프롬프트 다시 그리기, 그리고 mock LLM 노드와 chat 노드를 실제 ROS 토픽으로 연결한 왕복 테스트 |

노드 테스트는 Timer 콜백을 직접 호출하고 시계를 0.25초씩 진행합니다(이진 부동소수점으로 정확히 표현되어
경계 판정이 흔들리지 않음). 실제 10 Hz 주기는 Timer 설정값으로 따로 검사합니다.
