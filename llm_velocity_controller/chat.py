"""Interactive terminal client: type movement commands like a chat.

Each line is published on COMMAND_TOPIC; the node's progress events on
STATUS_TOPIC are printed in the same terminal. Quitting (exit, Ctrl+C,
Ctrl+D) sends a STOP command first.
"""

import json
import os
import signal
import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

try:
    import readline  # line editing and history for input()
except ImportError:  # pragma: no cover - always available on Ubuntu
    readline = None

from llm_velocity_controller.velocity_node import COMMAND_TOPIC
from llm_velocity_controller.velocity_node import STATUS_TOPIC

PROMPT = '> '
EXIT_WORDS = frozenset({'exit', 'quit', 'q', '종료', '나가기'})
STOP_ON_EXIT = '정지'
DISCOVERY_TIMEOUT_SEC = 3.0

BANNER = f"""LLM 속도 명령 채팅 (입력 → {COMMAND_TOPIC}, 상태 ← {STATUS_TOPIC})
  예) 앞으로 1초 동안 0.2m/s로 이동해 / 왼쪽으로 회전해 / 정지
  종료: exit, Ctrl+C, Ctrl+D (종료할 때 정지 명령을 보냅니다)"""


def parse_status(data):
    """Decode one STATUS_TOPIC payload; None if it is not a JSON object with a state."""
    try:
        event = json.loads(data)
    except (TypeError, ValueError):
        return None
    if not isinstance(event, dict) or 'state' not in event:
        return None
    return event


def format_status(event):
    """Turn one status event into a line for the terminal."""
    tag = f'[#{event.get("id", "?")}]'
    state = event['state']
    if state == 'querying':
        return f'  {tag} LLM 해석 중...'
    if state == 'stopped':
        return f'  {tag} 정지 (LLM 없이 즉시 처리)'
    if state == 'executing':
        return (f'  {tag} 실행: 직진 {event.get("linear_x", 0.0):+.2f} m/s, '
                f'회전 {event.get("angular_z", 0.0):+.2f} rad/s, '
                f'{event.get("duration", 0.0):.1f}초 동안')
    if state == 'done':
        return f'  {tag} 완료 (0속도 발행)'
    if state == 'llm_stop':
        return f'  {tag} LLM이 정지로 해석함 (불명확하거나 지원하지 않는 명령)'
    if state == 'rejected':
        return f'  {tag} 거부됨, 0속도 발행: {event.get("detail", "이유 없음")}'
    if state == 'stale':
        return f'  {tag} 이전 명령에 대한 늦은 응답이라 무시함'
    if state == 'shutdown':
        return '  velocity_node 종료 (0속도 발행)'
    return f'  {tag} {state}'


def is_exit_command(line):
    return line.strip().lower() in EXIT_WORDS


class ConsolePrinter:
    """Prints lines from background threads without mangling the line being typed."""

    def __init__(self, stream=sys.stdout, interactive=None):
        self._stream = stream
        self._interactive = stream.isatty() if interactive is None else interactive
        self._lock = threading.Lock()

    def stop_prompt(self):
        """After input() has ended, print plain lines instead of redrawing the prompt."""
        with self._lock:
            self._interactive = False

    def print(self, line):
        with self._lock:
            if self._interactive and readline is not None:
                typed = readline.get_line_buffer()
                # Clear the prompt line, print the event, then redraw prompt + typed text.
                self._stream.write(f'\r\033[K{line}\n{PROMPT}{typed}')
            else:
                self._stream.write(f'{line}\n')
            self._stream.flush()


class ChatNode(Node):

    def __init__(self, printer):
        super().__init__('llm_chat')
        # Set by the launch file: when velocity_node stops, close this window too.
        self.declare_parameter('exit_on_controller_shutdown', False)
        self._printer = printer
        self._publisher = self.create_publisher(String, COMMAND_TOPIC, 10)
        self._subscription = self.create_subscription(
            String, STATUS_TOPIC, self._on_status, 20)

    def _on_status(self, msg):
        event = parse_status(msg.data)
        if event is None:
            return
        self._printer.print(format_status(event))
        if (event['state'] == 'shutdown'
                and self.get_parameter('exit_on_controller_shutdown').value):
            self._printer.print('  velocity_node가 종료돼서 chat도 닫아요.')
            # input() blocks the main thread; interrupt it the same way Ctrl+C does.
            os.kill(os.getpid(), signal.SIGINT)

    def wait_for_controller(self, timeout_sec=DISCOVERY_TIMEOUT_SEC):
        """True once velocity_node subscribes to the command topic."""
        deadline = time.monotonic() + timeout_sec
        while self._publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def send(self, text):
        self._publisher.publish(String(data=text))


def main(args=None):
    # Ctrl+C must reach input() as KeyboardInterrupt with the context still
    # alive, so the STOP sent on exit can be published.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    printer = ConsolePrinter()
    node = ChatNode(printer)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name='chat-spin', daemon=True)
    spin_thread.start()

    print(BANNER)
    connected = node.wait_for_controller()
    if not connected:
        print('  [경고] velocity_node가 보이지 않아요. 다른 터미널에서 먼저 실행하세요:\n'
              '         ros2 run llm_velocity_controller velocity_node')
    echo_input = not sys.stdin.isatty()
    # Piped input: echo each line ourselves instead of printing a bare prompt.
    prompt = '' if echo_input else PROMPT
    try:
        while True:
            try:
                line = input(prompt)
            except EOFError:
                break
            if echo_input:  # piped input is not shown by the terminal
                printer.print(f'{PROMPT}{line}')
            line = line.strip()
            if not line:
                continue
            if is_exit_command(line):
                break
            if not connected:
                connected = node.wait_for_controller(timeout_sec=1.0)
            node.send(line)
    except KeyboardInterrupt:
        print()
    finally:
        printer.stop_prompt()
        node.send(STOP_ON_EXIT)
        print('  종료하면서 정지 명령을 보냈어요.')
        time.sleep(0.3)  # let the stop and its status reach the terminal
        executor.shutdown()
        # Destroying the node while spin() still runs aborts the process.
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
