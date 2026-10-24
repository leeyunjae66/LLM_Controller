import io
import json
import socket
import urllib.error

import pytest

from llm_velocity_controller.llm_client import LLMConnectionError
from llm_velocity_controller.llm_client import LLMResponseError
from llm_velocity_controller.llm_client import LLMTimeoutError
from llm_velocity_controller.llm_client import MockLLMClient
from llm_velocity_controller.llm_client import OllamaClient
from llm_velocity_controller.prompt import build_messages
from llm_velocity_controller.prompt import is_stop_command
from llm_velocity_controller.prompt import RESPONSE_SCHEMA
from llm_velocity_controller.prompt import SYSTEM_PROMPT
from llm_velocity_controller.validator import validate
from llm_velocity_controller.validator import ValidationError


@pytest.mark.parametrize('text', [
    '정지해', '멈춰!', '그만', '비상 정지', 'STOP', 'stop now', '', '   ', None])
def test_stop_commands_bypass_llm(text):
    assert is_stop_command(text)


@pytest.mark.parametrize('text', [
    '앞으로 1초 동안 0.2m/s로 이동해', '왼쪽으로 회전해', 'go forward'])
def test_motion_commands_go_to_llm(text):
    assert not is_stop_command(text)


def test_messages_and_prompt_contract():
    messages = build_messages('  앞으로 가  ')
    assert messages[0]['role'] == 'system'
    assert messages[1] == {'role': 'user', 'content': '앞으로 가'}
    for needle in ('JSON', 'Never clamp', '0.1', '0.3', 'linear_x', 'angular_z', 'duration'):
        assert needle in SYSTEM_PROMPT
    assert RESPONSE_SCHEMA['additionalProperties'] is False
    assert set(RESPONSE_SCHEMA['required']) == {'linear_x', 'angular_z', 'duration'}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _client(behaviour, **kwargs):
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        if isinstance(behaviour, BaseException):
            raise behaviour
        return _Response(behaviour.encode('utf-8'))
    return OllamaClient(opener=opener, **kwargs), calls


def test_request_body_and_content_extraction():
    reply = json.dumps({'message': {'role': 'assistant',
                                    'content': '{"linear_x":0.2,"angular_z":0,"duration":1}'}})
    logged = []
    client, calls = _client(reply, timeout_sec=7, raw_logger=logged.append,
                            model='m', api_url='http://example.invalid/api/chat')
    assert validate(client.complete('앞으로')).linear_x == 0.2
    request, timeout = calls[0]
    body = json.loads(request.data)
    assert timeout == 7
    assert request.full_url == 'http://example.invalid/api/chat'
    assert body['model'] == 'm'
    assert body['stream'] is False
    assert body['think'] is False
    assert body['format'] == RESPONSE_SCHEMA
    assert body['keep_alive'] == '30m'
    assert logged == [reply]


def test_warm_up_loads_model_without_generating():
    client, calls = _client('{"model": "m", "done": true}', model='m', keep_alive='5m')
    client.warm_up(timeout_sec=9)
    request, timeout = calls[0]
    assert json.loads(request.data) == {'model': 'm', 'messages': [], 'keep_alive': '5m'}
    assert timeout == 9


@pytest.mark.parametrize('error', [
    socket.timeout('timed out'),
    TimeoutError(),
    urllib.error.URLError(socket.timeout('timed out')),
])
def test_timeout_is_reported(error):
    client, _ = _client(error, timeout_sec=0.5)
    with pytest.raises(LLMTimeoutError, match='0.5'):
        client.complete('앞으로')


def test_connection_failure_is_reported():
    client, _ = _client(urllib.error.URLError(ConnectionRefusedError(111, 'refused')))
    with pytest.raises(LLMConnectionError, match='cannot reach'):
        client.complete('앞으로')


def test_http_error_is_reported():
    error = urllib.error.HTTPError('u', 404, 'nf', {}, io.BytesIO(b'model not found'))
    client, _ = _client(error)
    with pytest.raises(LLMResponseError, match='404.*model not found'):
        client.complete('앞으로')


@pytest.mark.parametrize('body', [
    'not json', '[]', '{"error": "boom"}', '{"message": {}}',
    '{"message": {"content": ""}}', '{"message": "x"}'])
def test_malformed_ollama_body_is_reported(body):
    client, _ = _client(body)
    with pytest.raises(LLMResponseError):
        client.complete('앞으로')


def test_constructor_rejects_bad_timeout():
    with pytest.raises(ValueError):
        OllamaClient(timeout_sec=0)


@pytest.mark.parametrize('text, expected', [
    ('앞으로 1초 동안 0.2m/s로 이동해', (0.2, 0.0, 1.0)),
    ('뒤로 1초 동안 0.1m/s로 이동해', (-0.1, 0.0, 1.0)),
    ('왼쪽으로 1초 동안 0.3rad/s로 회전해', (0.0, 0.3, 1.0)),
    ('오른쪽으로 2초 동안 회전해', (0.0, -0.3, 2.0)),
    ('노래 불러줘', (0.0, 0.0, 0.0)),
])
def test_mock_client_follows_prompt_contract(text, expected):
    command = validate(MockLLMClient().complete(text))
    assert (command.linear_x, command.angular_z, command.duration) == expected


def test_mock_client_out_of_range_is_rejected_by_validator():
    with pytest.raises(ValidationError, match='linear_x'):
        validate(MockLLMClient().complete('앞으로 10m/s로 이동해'))
