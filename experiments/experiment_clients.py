"""LLM backends for the experiments: the package's OllamaClient and a mock.

Both expose ask(system_prompt, schema, command) -> raw text from the model.
"""

import json
import random
import re
import urllib.error
import urllib.request

from llm_velocity_controller.llm_client import LLMConnectionError
from llm_velocity_controller.llm_client import LLMError
from llm_velocity_controller.llm_client import OllamaClient


class ExperimentOllamaClient(OllamaClient):
    """OllamaClient with a per-call system prompt / schema and tunable sampling.

    Transport, timeouts and error types are inherited unchanged from the
    package so experiment failures look exactly like node failures.
    """

    def __init__(self, temperature=0.0, seed=None, **kwargs):
        super().__init__(raw_logger=self._remember_envelope, **kwargs)
        self.temperature = float(temperature)
        self.seed = seed
        self._request = None
        self.last_envelope = None

    def _remember_envelope(self, raw):
        try:
            self.last_envelope = json.loads(raw)
        except ValueError:
            self.last_envelope = None

    def build_request_body(self, user_command):
        system_prompt, schema = self._request
        options = {'temperature': self.temperature}
        if self.seed is not None:
            options['seed'] = int(self.seed)
        return {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_command.strip()},
            ],
            'stream': False,
            'think': False,
            'format': schema,
            'keep_alive': self.keep_alive,
            'options': options,
        }

    def ask(self, system_prompt, schema, command):
        self._request = (system_prompt, schema)
        self.last_envelope = None
        return self.complete(command)

    def server_timing(self):
        """Ollama-reported durations (s) of the last call, if any."""
        env = self.last_envelope or {}
        timing = {}
        for key in ('total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration'):
            if isinstance(env.get(key), (int, float)):
                timing[key + '_sec'] = env[key] / 1e9
        for key in ('prompt_eval_count', 'eval_count'):
            if isinstance(env.get(key), int):
                timing[key] = env[key]
        return timing


def preflight(api_url, model, timeout_sec=5.0):
    """Check the server and the model before a long run. Raises LLMError with a hint."""
    base = api_url.split('/api/')[0]
    try:
        with urllib.request.urlopen(base + '/api/tags', timeout=timeout_sec) as response:
            tags = json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, OSError) as error:
        raise LLMConnectionError(
            f'Ollama 서버({base})에 연결할 수 없습니다: {getattr(error, "reason", error)}\n'
            '  → 다른 터미널에서 `ollama serve`를 실행하세요.') from None
    except ValueError:
        raise LLMError(f'{base}/api/tags 응답이 JSON이 아닙니다. Ollama 서버가 맞는지 확인하세요.') \
            from None
    names = {m.get('name', '') for m in tags.get('models', [])}
    names |= {n.split(':')[0] for n in names if n.endswith(':latest')}
    if model not in names:
        installed = ', '.join(sorted(names)) or '(없음)'
        raise LLMError(f"모델 '{model}'이 설치되어 있지 않습니다. 설치된 모델: {installed}\n"
                       f'  → `ollama pull {model}`')


class MockExperimentClient:
    """Deterministic, rule-based stand-in used to exercise the pipeline offline.

    Its numbers are NOT an LLM result and must never be reported as one. It
    adds seeded noise and occasional malformed replies so that statistics,
    error handling and plots all get exercised.
    """

    def __init__(self, seed=0, error_rate=0.05, noise=0.1):
        self._rng = random.Random(seed)
        self.error_rate = float(error_rate)
        self.noise = float(noise)

    def server_timing(self):
        return {}

    def _jitter(self, value):
        return round(value * (1.0 + self._rng.uniform(-self.noise, self.noise)), 3)

    def _speed_scale(self, text):
        for words, scale in ((('최대한',), 1.0), (('아주 빠르게',), 0.9), (('조금 빠르게',), 0.6),
                             (('빠르게',), 0.75), (('아주 천천히',), 0.15),
                             (('천천히', '살살', '느긋', '조심'), 0.3)):
            if any(w in text for w in words):
                return scale
        return 0.5

    def _single(self, text):
        if '계속' in text or '때까지' in text:
            return {'linear_x': 0, 'angular_z': 0, 'duration': 0}
        distance = None
        for words, metres in ((('한 뼘',), 0.2), (('두 뼘',), 0.4), (('한 발짝',), 0.6),
                              (('두 걸음',), 1.3), (('팔',), 0.7), (('손바닥',), 0.18),
                              (('아주 조금',), 0.1), (('책상',), 1.2), (('꽤 멀리',), 1.8),
                              (('조금',), 0.3)):
            if any(w in text for w in words):
                distance = metres
                break
        duration = 3.0
        for words, seconds in ((('아주 잠깐',), 1.0), (('잠깐', '잠시'), 2.0),
                               (('한동안',), 6.0), (('오래',), 8.0)):
            if any(w in text for w in words):
                duration = seconds
                break
        if '유턴' in text:
            w = 0.5 if '왼' in text else -0.5
            return {'linear_x': 0.2 if '앞' in text else 0.0, 'angular_z': w,
                    'duration': round(3.14159 / abs(w), 3)}
        if '원' in text:
            w = -0.3 if '오른' in text else 0.3
            v = 0.1 if '작은' in text else 0.4 if '큰' in text else 0.2
            return {'linear_x': self._jitter(v), 'angular_z': w, 'duration': 8.0}
        if any(w in text for w in ('포물선', '사각형', '지그재그')):
            return {'linear_x': 0, 'angular_z': 0, 'duration': 0}
        if '회전' in text:
            rate = self._jitter(0.5 * self._speed_scale(text))
            return {'linear_x': 0, 'angular_z': rate if '왼' in text else -rate,
                    'duration': duration}
        speed = self._jitter(0.5 * self._speed_scale(text))
        if distance is not None:
            return {'linear_x': speed, 'angular_z': 0, 'duration': round(distance / speed, 3)}
        return {'linear_x': -speed if '뒤' in text else speed, 'angular_z': 0,
                'duration': duration}

    def _actions(self, text):
        if '정사각형' in text or '직사각형' in text:
            sides = (1.0, 1.0) if '정사각형' in text else (1.5, 0.8)
            actions = []
            for i in range(4):
                actions.append({'linear_x': 0.2, 'angular_z': 0,
                                'duration': self._jitter(sides[i % 2] / 0.2)})
                actions.append({'linear_x': 0, 'angular_z': 0.5, 'duration': 3.1416})
            return {'actions': actions}
        if '지그재그' in text:
            return {'actions': [{'linear_x': 0.2, 'angular_z': (0.4 if i % 2 else -0.4),
                                 'duration': 2.0} for i in range(6)]}
        if '포물선' in text:
            return {'actions': [{'linear_x': 0.2, 'angular_z': 0.1 * k, 'duration': 2.0}
                                for k in (-2, -1, 0, 1, 2)]}
        single = self._single(text)
        return {'actions': [single] if single['duration'] > 0 else []}

    def ask(self, system_prompt, schema, command):
        roll = self._rng.random()
        if roll < self.error_rate / 2:
            return 'Sure! Here is the command: {"linear_x": 0.1'
        if roll < self.error_rate:
            return json.dumps({'linear_x': 'fast', 'angular_z': 0, 'duration': 1})
        text = re.sub(r'\s+', ' ', command)
        reply = self._actions(text) if 'actions' in schema.get('properties', {}) \
            else self._single(text)
        return json.dumps(reply, ensure_ascii=False)
