import io
import json
import threading
import time

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor

from llm_velocity_controller import chat
from llm_velocity_controller.chat import ChatNode
from llm_velocity_controller.chat import ConsolePrinter
from llm_velocity_controller.chat import format_status
from llm_velocity_controller.chat import is_exit_command
from llm_velocity_controller.chat import parse_status
from llm_velocity_controller.llm_client import MockLLMClient
from llm_velocity_controller.velocity_node import STATUS_STATES
from llm_velocity_controller.velocity_node import VelocityNode


@pytest.mark.parametrize('data', ['', 'not json', '[]', '{"id": 1}', None])
def test_parse_status_rejects_garbage(data):
    assert parse_status(data) is None


def test_parse_status_keeps_korean_text():
    event = parse_status(json.dumps({'id': 3, 'state': 'querying', 'text': '앞으로'},
                                    ensure_ascii=False))
    assert event == {'id': 3, 'state': 'querying', 'text': '앞으로'}


@pytest.mark.parametrize('state', STATUS_STATES)
def test_every_node_state_has_a_message(state):
    line = format_status({'id': 7, 'state': state})
    assert line.strip()
    assert line != f'  [#7] {state}'          # not the unknown-state fallback


def test_format_executing_shows_values():
    line = format_status({'id': 2, 'state': 'executing',
                          'linear_x': -0.1, 'angular_z': 0.0, 'duration': 1.0})
    assert line == '  [#2] 실행: 직진 -0.10 m/s, 회전 +0.00 rad/s, 1.0초 동안'


def test_format_rejected_shows_reason():
    line = format_status({'id': 4, 'state': 'rejected',
                          'detail': 'ValidationError: linear_x 10.0 outside [-0.2, 0.2] m/s'})
    assert 'linear_x 10.0' in line and '0속도' in line


def test_format_unknown_state_falls_back():
    assert format_status({'id': 9, 'state': 'new_thing'}) == '  [#9] new_thing'


@pytest.mark.parametrize('line, expected', [
    ('exit', True), (' QUIT ', True), ('q', True), ('종료', True), ('나가기', True),
    ('앞으로 가', False), ('정지', False), ('', False)])
def test_exit_words(line, expected):
    assert is_exit_command(line) is expected


def test_printer_plain_stream():
    stream = io.StringIO()
    ConsolePrinter(stream, interactive=False).print('hello')
    assert stream.getvalue() == 'hello\n'


def test_printer_redraws_prompt_and_typed_text(monkeypatch):
    class FakeReadline:
        @staticmethod
        def get_line_buffer():
            return '앞으로 1'
    monkeypatch.setattr(chat, 'readline', FakeReadline)
    stream = io.StringIO()
    ConsolePrinter(stream, interactive=True).print('  [#1] 완료')
    assert stream.getvalue() == '\r\033[K  [#1] 완료\n> 앞으로 1'


def test_printer_stops_redrawing_prompt_after_exit(monkeypatch):
    monkeypatch.setattr(chat, 'readline', object())
    stream = io.StringIO()
    printer = ConsolePrinter(stream, interactive=True)
    printer.stop_prompt()
    printer.print('  [#2] 정지')
    assert stream.getvalue() == '  [#2] 정지\n'


class ListPrinter:
    def __init__(self):
        self.lines = []

    def print(self, line):
        self.lines.append(line)


def _status_msg(state):
    from std_msgs.msg import String
    return String(data=json.dumps({'id': 5, 'state': state}))


@pytest.mark.parametrize('flag, expect_exit', [(False, False), (True, True)])
def test_chat_closes_on_controller_shutdown_only_when_asked(monkeypatch, flag, expect_exit):
    from rclpy.parameter import Parameter
    killed = []
    monkeypatch.setattr(chat.os, 'kill', lambda pid, sig: killed.append(sig))
    rclpy.init()
    try:
        printer = ListPrinter()
        node = ChatNode(printer)
        node.set_parameters([Parameter('exit_on_controller_shutdown', value=flag)])
        node._on_status(_status_msg('done'))
        assert killed == []
        node._on_status(_status_msg('shutdown'))
        assert (killed == [chat.signal.SIGINT]) is expect_exit
        assert any('종료' in line for line in printer.lines)
        node.destroy_node()
    finally:
        rclpy.shutdown()


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_chat_round_trip_through_ros_topics():
    """chat -> /llm/user_command -> VelocityNode(mock LLM) -> /llm/status -> chat."""
    rclpy.init()
    try:
        printer = ListPrinter()
        controller = VelocityNode(llm_client=MockLLMClient())
        client = ChatNode(printer)
        executor = SingleThreadedExecutor()
        executor.add_node(controller)
        executor.add_node(client)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()
        try:
            assert client.wait_for_controller(timeout_sec=10.0)
            # status subscription must be matched too, or early events are lost
            assert _wait_for(lambda: controller._status_publisher.get_subscription_count() > 0)

            client.send('앞으로 1초 동안 0.2m/s로 이동해')
            assert _wait_for(lambda: any('완료' in line for line in printer.lines))
            client.send('앞으로 10m/s로 이동해')
            assert _wait_for(lambda: any('거부됨' in line for line in printer.lines))
            client.send('정지')
            assert _wait_for(lambda: any('정지 (LLM 없이' in line for line in printer.lines))

            joined = '\n'.join(printer.lines)
            assert '[#1] LLM 해석 중...' in joined
            assert '[#1] 실행: 직진 +0.20 m/s, 회전 +0.00 rad/s, 1.0초 동안' in joined
            assert '[#1] 완료' in joined
            assert '[#2] 거부됨' in joined and 'linear_x 10.0' in joined
            assert '[#3] 정지' in joined
        finally:
            executor.shutdown()
            spin.join(5.0)
            client.destroy_node()
            controller.destroy_node()
    finally:
        rclpy.shutdown()
