import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import ExecuteProcess
from launch.actions import LogInfo
from launch.actions import Shutdown
from launch.actions import TimerAction
from launch_ros.actions import Node
import pytest

LAUNCH_FILE = Path(__file__).resolve().parent.parent / 'launch' / 'llm_velocity_controller.launch.py'


@pytest.fixture(scope='module')
def launch_module():
    spec = importlib.util.spec_from_file_location('llm_launch', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(module, **overrides):
    context = LaunchContext()
    for name, default, _ in module.ARGUMENTS:
        context.launch_configurations[name] = overrides.get(name, default)
    return context


def _executables(actions):
    nodes = []
    for action in actions:
        if isinstance(action, TimerAction):
            nodes += _executables(action.actions)
        elif isinstance(action, Node):
            nodes.append(action.node_executable)
    return nodes


@pytest.mark.parametrize('value, expected', [
    ('true', True), ('True', True), ('1', True), ('on', True),
    ('false', False), ('0', False), ('no', False)])
def test_as_bool(launch_module, value, expected):
    assert launch_module.as_bool(value) is expected


def test_as_bool_rejects_typos(launch_module):
    with pytest.raises(ValueError):
        launch_module.as_bool('ture')


def test_ollama_base_url(launch_module):
    assert launch_module.ollama_base_url('http://localhost:11434/api/chat') == \
        'http://localhost:11434'
    assert launch_module.ollama_base_url('http://10.0.0.5:8080/api/chat') == \
        'http://10.0.0.5:8080'
    with pytest.raises(ValueError):
        launch_module.ollama_base_url('localhost:11434')


def test_ollama_running_is_false_when_nothing_listens(launch_module):
    assert launch_module.ollama_running('http://127.0.0.1:9/api/chat', timeout_sec=0.5) is False


def test_declares_documented_arguments(launch_module):
    description = launch_module.generate_launch_description()
    names = {action.name for action in description.entities if hasattr(action, 'default_value')}
    assert names == {name for name, _, _ in launch_module.ARGUMENTS}


def test_mock_backend_starts_node_and_chat_without_ollama(launch_module):
    actions = launch_module._launch_setup(_context(launch_module, llm_backend='mock'))
    assert _executables(actions) == ['velocity_node', 'chat']
    assert not any(isinstance(a, ExecuteProcess) and not isinstance(a, Node) for a in actions)


def test_chat_window_uses_terminal_prefix_and_shuts_down_launch(launch_module):
    actions = launch_module._launch_setup(_context(launch_module, llm_backend='mock'))
    chat = [a for a in actions if isinstance(a, Node) and a.node_executable == 'chat'][0]
    prefix = ''.join(part.perform(LaunchContext())
                     for part in chat.process_description.prefix)
    assert prefix == launch_module.DEFAULT_CHAT_TERMINAL
    assert any(isinstance(action, Shutdown) for action in chat.get_sub_entities())
    assert chat._Node__parameters, 'chat must get exit_on_controller_shutdown'


def test_chat_can_be_disabled(launch_module):
    actions = launch_module._launch_setup(_context(launch_module, llm_backend='mock', chat='false'))
    assert _executables(actions) == ['velocity_node']


def test_existing_ollama_is_reused(launch_module, monkeypatch):
    monkeypatch.setattr(launch_module, 'ollama_running', lambda url: True)
    actions = launch_module._launch_setup(_context(launch_module, chat='false'))
    assert isinstance(actions[0], LogInfo)
    assert isinstance(actions[1], Node)          # started right away, no delay


def test_missing_ollama_is_started_before_node(launch_module, monkeypatch):
    monkeypatch.setattr(launch_module, 'ollama_running', lambda url: False)
    monkeypatch.setattr(launch_module.shutil, 'which', lambda name: '/usr/local/bin/ollama')
    actions = launch_module._launch_setup(_context(launch_module, chat='false'))
    serve = [a for a in actions if isinstance(a, ExecuteProcess) and not isinstance(a, Node)]
    assert len(serve) == 1
    assert [part.perform(LaunchContext()) for parts in serve[0].cmd for part in parts] == \
        ['ollama', 'serve']
    timers = [a for a in actions if isinstance(a, TimerAction)]
    assert len(timers) == 1 and _executables(timers) == ['velocity_node']


def test_start_ollama_false_never_starts_server(launch_module, monkeypatch):
    monkeypatch.setattr(launch_module, 'ollama_running', lambda url: False)
    actions = launch_module._launch_setup(
        _context(launch_module, chat='false', start_ollama='false'))
    assert not any(isinstance(a, ExecuteProcess) and not isinstance(a, Node) for a in actions)
    assert _executables(actions) == ['velocity_node']


def test_invalid_start_ollama_value_is_rejected(launch_module, monkeypatch):
    monkeypatch.setattr(launch_module, 'ollama_running', lambda url: False)
    with pytest.raises(ValueError, match='start_ollama'):
        launch_module._launch_setup(_context(launch_module, start_ollama='yes'))
