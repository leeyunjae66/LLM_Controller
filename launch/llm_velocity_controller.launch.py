"""Start everything at once: Ollama (if needed), velocity_node and the chat client.

`ros2 launch` does not forward keyboard input to its processes, so the chat
client is opened in its own terminal window (`chat_terminal`). Closing that
window (or typing exit) shuts the whole launch down when `shutdown_with_chat`
is true.

    ros2 launch llm_velocity_controller llm_velocity_controller.launch.py
    ros2 launch llm_velocity_controller llm_velocity_controller.launch.py llm_backend:=mock
    ros2 launch llm_velocity_controller llm_velocity_controller.launch.py chat:=false
    ros2 launch llm_velocity_controller llm_velocity_controller.launch.py cmd_vel_topic:=/rubi_gazebo/cmd_vel
"""

import shutil
import urllib.error
import urllib.parse
import urllib.request

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import LogInfo
from launch.actions import OpaqueFunction
from launch.actions import Shutdown
from launch.actions import TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = 'llm_velocity_controller'
DEFAULT_API_URL = 'http://localhost:11434/api/chat'
# --wait keeps the gnome-terminal client alive until the window closes, so the
# launch can tell when the chat window is gone.
DEFAULT_CHAT_TERMINAL = 'gnome-terminal --wait --title=LLM-chat --'
OLLAMA_STARTUP_DELAY_SEC = 3.0

ARGUMENTS = (
    ('llm_backend', 'ollama', "'ollama' or 'mock' (no LLM needed)"),
    ('model', 'qwen3:4b', 'Ollama model name'),
    ('api_url', DEFAULT_API_URL, 'Ollama /api/chat endpoint'),
    ('timeout_sec', '30.0', 'LLM request timeout in seconds'),
    ('keep_alive', '30m', 'how long Ollama keeps the model loaded'),
    ('log_raw_response', 'true', 'log the raw LLM reply in velocity_node'),
    ('start_ollama', 'auto',
     "'auto': start `ollama serve` only if nothing answers at api_url; 'false': never"),
    ('chat', 'true', 'open the chat client in a new terminal window'),
    ('chat_terminal', DEFAULT_CHAT_TERMINAL,
     "command prefix that opens a terminal, e.g. 'terminator -x'"),
    ('shutdown_with_chat', 'true', 'stop everything when the chat window closes'),
    ('cmd_vel_topic', '/cmd_vel',
     "where velocity_node's /cmd_vel goes, e.g. '/rubi_gazebo/cmd_vel'"),
)


def as_bool(value):
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'expected true/false, got {value!r}')


def ollama_base_url(api_url):
    """'http://host:11434/api/chat' -> 'http://host:11434'."""
    parsed = urllib.parse.urlparse(api_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f'api_url must look like http://host:port/api/chat, got {api_url!r}')
    return f'{parsed.scheme}://{parsed.netloc}'


def ollama_running(api_url, timeout_sec=1.0):
    try:
        with urllib.request.urlopen(ollama_base_url(api_url) + '/api/version',
                                    timeout=timeout_sec) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _ollama_actions(api_url, start_ollama):
    """Return (actions, started) for making sure an Ollama server is reachable."""
    base = ollama_base_url(api_url)
    if start_ollama == 'false':
        return [LogInfo(msg=f'start_ollama:=false, expecting Ollama at {base}')], False
    if start_ollama != 'auto':
        raise ValueError(f"start_ollama must be 'auto' or 'false', got {start_ollama!r}")
    if ollama_running(api_url):
        return [LogInfo(msg=f'Ollama already running at {base}')], False
    if shutil.which('ollama') is None:
        return [LogInfo(msg=f'[WARN] nothing answers at {base} and `ollama` is not '
                            'installed; commands will be rejected until it is up')], False
    serve = ExecuteProcess(
        cmd=['ollama', 'serve'], name='ollama', output='log',
        additional_env={'OLLAMA_HOST': urllib.parse.urlparse(base).netloc})
    return [LogInfo(msg=f'starting `ollama serve` at {base}'), serve], True


def _launch_setup(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    backend = arg('llm_backend')
    actions = []
    ollama_started = False
    if backend == 'ollama':
        ollama, ollama_started = _ollama_actions(arg('api_url'), arg('start_ollama'))
        actions += ollama

    controller = Node(
        package=PACKAGE, executable='velocity_node', output='screen', emulate_tty=True,
        parameters=[{
            'llm_backend': backend,
            'model': arg('model'),
            'api_url': arg('api_url'),
            'timeout_sec': float(arg('timeout_sec')),
            'keep_alive': arg('keep_alive'),
            'log_raw_response': as_bool(arg('log_raw_response')),
        }],
        remappings=[('/cmd_vel', arg('cmd_vel_topic'))])
    # Give a freshly started server a moment so the node's warm-up can connect.
    actions.append(TimerAction(period=OLLAMA_STARTUP_DELAY_SEC, actions=[controller])
                   if ollama_started else controller)

    if as_bool(arg('chat')):
        on_exit = ([Shutdown(reason='chat window closed')]
                   if as_bool(arg('shutdown_with_chat')) else None)
        actions.append(Node(
            package=PACKAGE, executable='chat', prefix=arg('chat_terminal'),
            output='screen', on_exit=on_exit,
            # Ctrl+C on the launch cannot reach a separate terminal window,
            # so chat closes itself when velocity_node reports shutdown.
            parameters=[{'exit_on_controller_shutdown': True}]))
    return actions


def generate_launch_description():
    declared = [DeclareLaunchArgument(name, default_value=default, description=description)
                for name, default, description in ARGUMENTS]
    return LaunchDescription(declared + [OpaqueFunction(function=_launch_setup)])
