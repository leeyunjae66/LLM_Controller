"""Ollama /api/chat client (standard library only)."""

import json
import re
import socket
import time
import urllib.error
import urllib.request

from llm_velocity_controller.prompt import build_messages
from llm_velocity_controller.prompt import DEFAULT_ANGULAR_SPEED
from llm_velocity_controller.prompt import DEFAULT_DURATION
from llm_velocity_controller.prompt import DEFAULT_LINEAR_SPEED
from llm_velocity_controller.prompt import RESPONSE_SCHEMA

DEFAULT_MODEL = 'qwen3:4b'
DEFAULT_API_URL = 'http://localhost:11434/api/chat'
DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_KEEP_ALIVE = '30m'   # keep the model loaded between commands


class LLMError(RuntimeError):
    """Base class for every LLM failure; the caller must stop the robot."""


class LLMConnectionError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMResponseError(LLMError):
    pass


def _is_timeout(error):
    reason = getattr(error, 'reason', error)
    return isinstance(reason, (socket.timeout, TimeoutError)) or 'timed out' in str(reason)


class OllamaClient:
    """Sends one command to Ollama and returns the raw JSON text of the reply."""

    def __init__(self, model=DEFAULT_MODEL, api_url=DEFAULT_API_URL,
                 timeout_sec=DEFAULT_TIMEOUT_SEC, keep_alive=DEFAULT_KEEP_ALIVE,
                 raw_logger=None, opener=None):
        if timeout_sec <= 0:
            raise ValueError('timeout_sec must be positive')
        self.model = model
        self.api_url = api_url
        self.timeout_sec = float(timeout_sec)
        self.keep_alive = keep_alive
        self._raw_logger = raw_logger
        self._open = opener or urllib.request.urlopen

    def build_request_body(self, user_command):
        return {
            'model': self.model,
            'messages': build_messages(user_command),
            'stream': False,
            'think': False,
            'format': RESPONSE_SCHEMA,
            'keep_alive': self.keep_alive,
            'options': {'temperature': 0},
        }

    def warm_up(self, timeout_sec=120.0):
        """Load the model into memory so the first real command is not delayed.

        An empty `messages` list makes Ollama load the model without generating.
        """
        body = {'model': self.model, 'messages': [], 'keep_alive': self.keep_alive}
        self._post(body, timeout_sec)

    def complete(self, user_command):
        raw = self._post(self.build_request_body(user_command), self.timeout_sec)
        if self._raw_logger is not None:
            self._raw_logger(raw)
        return self.extract_content(raw)

    def _post(self, payload, timeout_sec):
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            self.api_url, data=body, method='POST',
            headers={'Content-Type': 'application/json'})
        try:
            with self._open(request, timeout=timeout_sec) as response:
                return response.read().decode('utf-8')
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', 'replace')[:300]
            raise LLMResponseError(f'HTTP {error.code} from Ollama: {detail}') from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as error:
            if _is_timeout(error):
                raise LLMTimeoutError(
                    f'Ollama did not answer within {timeout_sec:.1f} s') from None
            raise LLMConnectionError(
                f'cannot reach Ollama at {self.api_url}: {getattr(error, "reason", error)}'
            ) from None

    @staticmethod
    def extract_content(raw):
        """Return message.content from an Ollama /api/chat response body."""
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as error:
            raise LLMResponseError(f'Ollama returned non-JSON body: {error}') from None
        if not isinstance(envelope, dict):
            raise LLMResponseError('Ollama response is not a JSON object')
        if 'error' in envelope:
            raise LLMResponseError(f'Ollama error: {envelope["error"]}')
        message = envelope.get('message')
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError('Ollama response has no message.content')
        return content


class MockLLMClient:
    """Rule-based stand-in for Ollama, for exercising the node without a model.

    It mimics the prompt's contract (stated values copied verbatim, defaults
    otherwise, unclear commands -> zeros) for the documented test sentences.
    """

    _SPEED = re.compile(r'(-?\d+(?:\.\d+)?)\s*(m/s|rad/s)', re.IGNORECASE)
    _SECONDS = re.compile(r'(\d+(?:\.\d+)?)\s*(초|s\b|sec)', re.IGNORECASE)

    def __init__(self, delay_sec=0.0, raw_logger=None):
        self.delay_sec = float(delay_sec)
        self._raw_logger = raw_logger

    def complete(self, user_command):
        if self.delay_sec > 0.0:
            time.sleep(self.delay_sec)
        text = user_command.strip()
        speed_match = self._SPEED.search(text)
        speed = float(speed_match.group(1)) if speed_match else None
        seconds_match = self._SECONDS.search(text)
        duration = float(seconds_match.group(1)) if seconds_match else DEFAULT_DURATION

        linear_x = angular_z = 0.0
        if '회전' in text or '돌아' in text:
            rate = speed if speed is not None else DEFAULT_ANGULAR_SPEED
            if '오른' in text:
                angular_z = -rate
            elif '왼' in text:
                angular_z = rate
            else:
                duration = 0.0
        elif '앞' in text or '전진' in text:
            linear_x = speed if speed is not None else DEFAULT_LINEAR_SPEED
        elif '뒤' in text or '후진' in text:
            linear_x = -(speed if speed is not None else DEFAULT_LINEAR_SPEED)
        else:
            duration = 0.0

        content = json.dumps(
            {'linear_x': linear_x, 'angular_z': angular_z, 'duration': duration})
        if self._raw_logger is not None:
            self._raw_logger(content)
        return content
