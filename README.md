# llm_velocity_controller

자연어 이동 명령을 로컬 LLM(Ollama + `qwen3:4b`)이 JSON 속도 명령으로 바꾸고,
Validator가 검증한 뒤 `geometry_msgs/msg/Twist`로 **테스트 토픽 `/llm/cmd_vel_test`에만**
발행하는 ROS 2 Humble 패키지입니다. 실제 `/cmd_vel`에는 절대 발행하지 않습니다(토픽 이름이 코드에 고정).

```text
/llm/user_command (std_msgs/String)
   │
   ├─ STOP 키워드(정지·멈춰·그만·비상·stop…) ──► 즉시 0속도 (LLM 호출 없음)
   │
   └─ Worker Thread: prompt.py → llm_client.py (Ollama /api/chat) → validator.py
                                          │ (generation, 결과|오류) 큐
                                          ▼
                     10 Hz ROS Timer ──► /llm/cmd_vel_test (geometry_msgs/Twist)
```

## 모듈

| 파일 | 역할 |
|---|---|
| `prompt.py` | System Prompt, 메시지 구성, Structured Output용 JSON Schema, 로컬 STOP 판별(`is_stop_command`) |
| `llm_client.py` | `OllamaClient`(표준 라이브러리 `urllib`), 연결/Timeout/응답 오류 예외, 원본 응답 로깅, warm-up. `MockLLMClient`(Ollama 없이 실행용) |
| `validator.py` | ROS 비의존 순수 Python 검증기. 범위 밖 값은 **보정하지 않고 거부** |
| `velocity_node.py` | `llm_velocity_controller` 노드. Worker Thread + 10 Hz Timer, generation 카운터 |

### JSON 계약과 허용 범위

```json
{"linear_x": 0.2, "angular_z": 0.0, "duration": 1.0}
```

| 필드 | 범위 | 비고 |
|---|---|---|
| `linear_x` | -0.2 ~ 0.2 m/s | 앞(+), 뒤(-) |
| `angular_z` | -0.5 ~ 0.5 rad/s | 왼쪽(+, 반시계), 오른쪽(-) |
| `duration` | 0 ~ 2.0 s | 음수 거부 |

거부 조건: JSON 파싱 실패(코드블록·꼬리 문자열 포함), 객체가 아님, 필드 누락, 추가 필드,
중복 키, 숫자가 아님(`true/false`, 문자열, `null`), `NaN`/`Infinity`/오버플로, 범위 초과.
명시되지 않은 값의 기본값: 전후진 0.1 m/s, 회전 0.3 rad/s, 지속시간 1초. 불명확한 명령은 정지(모두 0).

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

```bash
ros2 run llm_velocity_controller velocity_node
# Ollama 없이:
ros2 run llm_velocity_controller velocity_node --ros-args -p llm_backend:=mock
```

다른 터미널에서 명령 발행과 결과 확인:

```bash
ros2 topic pub --once /llm/user_command std_msgs/msg/String "{data: '앞으로 1초 동안 0.2m/s로 이동해'}"
ros2 topic echo /llm/cmd_vel_test
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
| `test/test_velocity_node.py` | 실제 LLM 없이 가짜 LLM·가짜 시계·발행 기록기로 노드 검증: duration 후 0, 즉시 STOP, 느린 추론 중 STOP, 명령 취소, 오래된 응답 폐기, Timeout/잘못된 JSON/범위 초과 시 0, 종료 시 0, 10 Hz 주기, 출력 토픽 고정 |

노드 테스트는 Timer 콜백을 직접 호출하고 시계를 0.25초씩 진행합니다(이진 부동소수점으로 정확히 표현되어
경계 판정이 흔들리지 않음). 실제 10 Hz 주기는 Timer 설정값으로 따로 검사합니다.
