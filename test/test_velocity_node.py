"""Node behaviour with a fake LLM, fake clock and recording publisher (no Ollama)."""

import json
import threading

import pytest
import rclpy
from std_msgs.msg import String

from llm_velocity_controller.llm_client import LLMTimeoutError
from llm_velocity_controller.velocity_node import OUTPUT_TOPIC
from llm_velocity_controller.velocity_node import VelocityNode


def reply(linear_x=0.0, angular_z=0.0, duration=0.0):
    return json.dumps({'linear_x': linear_x, 'angular_z': angular_z, 'duration': duration})


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, twist):
        self.messages.append(twist)

    def values(self):
        return [(m.linear.x, m.angular.z) for m in self.messages]

    def clear(self):
        self.messages.clear()


class FakeLLM:
    """Replies from a script; a reply may be an Exception or a threading.Event gate."""

    def __init__(self):
        self.script = {}
        self.gates = {}
        self.calls = []

    def complete(self, text):
        self.calls.append(text)
        gate = self.gates.get(text)
        if gate is not None:
            assert gate.wait(5.0), 'test gate never released'
        result = self.script[text]
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture(scope='module', autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def harness():
    llm, clock, recorder = FakeLLM(), FakeClock(), Recorder()
    node = VelocityNode(llm_client=llm, clock=clock)
    node._publisher = recorder
    yield node, llm, clock, recorder
    node.destroy_node()


def send(node, text):
    node._on_user_command(String(data=text))


# 0.25 s steps are exact in binary floating point, so tick counts at the
# duration boundary are deterministic. The real 10 Hz period is checked below.
STEP = 0.25


def tick(node, clock, seconds=STEP):
    node._on_timer()
    clock.now += seconds


def assert_other_fields_zero(recorder):
    for m in recorder.messages:
        assert (m.linear.y, m.linear.z, m.angular.x, m.angular.y) == (0.0, 0.0, 0.0, 0.0)


def test_output_topic_is_test_topic_only():
    assert OUTPUT_TOPIC == '/llm/cmd_vel_test'


def test_timer_runs_at_10_hz(harness):
    node = harness[0]
    assert node._timer.timer_period_ns == 100_000_000


def test_command_runs_for_duration_then_zero(harness):
    node, llm, clock, rec = harness
    llm.script['fwd'] = reply(0.2, 0.0, 1.0)
    send(node, 'fwd')
    node.wait_for_workers(5)
    assert rec.values() == [(0.0, 0.0)]          # zero on receipt
    rec.clear()
    for _ in range(8):
        tick(node, clock)
    moving = [v for v in rec.values() if v != (0.0, 0.0)]
    assert len(moving) == 4                     # 1.0 s / 0.25 s steps
    assert set(moving) == {(0.2, 0.0)}
    assert rec.values()[4] == (0.0, 0.0)        # zero right after the duration
    assert len(rec.values()) == 5               # idle afterwards: nothing more
    assert_other_fields_zero(rec)


def test_stop_is_immediate_and_skips_llm(harness):
    node, llm, clock, rec = harness
    llm.script['fwd'] = reply(0.2, 0.0, 2.0)
    send(node, 'fwd')
    node.wait_for_workers(5)
    tick(node, clock)
    tick(node, clock)
    rec.clear()
    send(node, '정지해')
    assert rec.values() == [(0.0, 0.0)]         # published inside the callback
    assert llm.calls == ['fwd']                 # STOP never reached the LLM
    for _ in range(5):
        tick(node, clock)
    assert rec.values() == [(0.0, 0.0)]


def test_stop_does_not_wait_for_slow_inference(harness):
    node, llm, clock, rec = harness
    llm.script['slow'] = reply(0.1, 0.0, 1.0)
    llm.gates['slow'] = threading.Event()
    send(node, 'slow')
    send(node, 'stop')                          # LLM still blocked
    assert rec.values() == [(0.0, 0.0), (0.0, 0.0)]
    llm.gates['slow'].set()
    node.wait_for_workers(5)
    rec.clear()
    for _ in range(5):
        tick(node, clock)
    assert rec.values() == []                   # late reply discarded


def test_new_command_cancels_previous(harness):
    node, llm, clock, rec = harness
    llm.script['fwd'] = reply(0.2, 0.0, 2.0)
    llm.script['left'] = reply(0.0, 0.3, 0.5)
    send(node, 'fwd')
    node.wait_for_workers(5)
    for _ in range(3):
        tick(node, clock)
    rec.clear()
    send(node, 'left')
    assert rec.values() == [(0.0, 0.0)]         # previous motion stopped at once
    node.wait_for_workers(5)
    for _ in range(10):
        tick(node, clock)
    values = rec.values()
    assert (0.2, 0.0) not in values
    assert values[1:3] == [(0.0, 0.3)] * 2     # 0.5 s
    assert values[3] == (0.0, 0.0)


def test_stale_response_cannot_override_latest(harness):
    node, llm, clock, rec = harness
    llm.script['old'] = reply(0.2, 0.0, 2.0)
    llm.script['new'] = reply(-0.1, 0.0, 1.0)
    llm.gates['old'] = threading.Event()
    send(node, 'old')                           # blocks in the LLM
    send(node, 'new')
    llm.gates['old'].set()                      # old reply arrives late
    node.wait_for_workers(5)
    rec.clear()
    for _ in range(12):
        tick(node, clock)
    assert (0.2, 0.0) not in rec.values()
    assert rec.values().count((-0.1, 0.0)) == 4


def test_late_reply_after_completion_of_newer_is_discarded(harness):
    node, llm, clock, rec = harness
    llm.script['old'] = reply(0.2, 0.0, 2.0)
    llm.script['new'] = reply(0.0, -0.3, 0.2)
    llm.gates['old'] = threading.Event()
    send(node, 'old')
    send(node, 'new')
    while len(llm.calls) < 2:
        pass
    for _ in range(5):                          # 'new' runs and finishes
        tick(node, clock)
    llm.gates['old'].set()
    node.wait_for_workers(5)
    rec.clear()
    for _ in range(5):
        tick(node, clock)
    assert rec.values() == []


@pytest.mark.parametrize('bad', [
    LLMTimeoutError('Ollama did not answer within 30.0 s'),
    ConnectionError('refused'),
    'not json',
    reply(10.0, 0.0, 1.0),                      # 앞으로 10m/s -> rejected
    '{"linear_x": 0.1, "angular_z": 0, "duration": 1, "extra": 1}',
    '{"linear_x": NaN, "angular_z": 0, "duration": 1}',
    reply(0.1, 0.0, -1.0),
])
def test_failures_publish_zero_and_cancel_motion(harness, bad):
    node, llm, clock, rec = harness
    llm.script['fwd'] = reply(0.2, 0.0, 2.0)
    llm.script['bad'] = bad
    send(node, 'fwd')
    node.wait_for_workers(5)
    tick(node, clock)
    send(node, 'bad')
    node.wait_for_workers(5)
    rec.clear()
    for _ in range(10):
        tick(node, clock)
    assert rec.values() == [(0.0, 0.0)]         # one zero, no motion resumes


def test_llm_stop_reply_publishes_zero(harness):
    node, llm, clock, rec = harness
    llm.script['sing'] = reply(0.0, 0.0, 0.0)
    send(node, 'sing')
    node.wait_for_workers(5)
    rec.clear()
    for _ in range(3):
        tick(node, clock)
    assert rec.values() == [(0.0, 0.0)]


def test_shutdown_publishes_zero_and_ignores_inflight(harness):
    node, llm, clock, rec = harness
    llm.script['fwd'] = reply(0.2, 0.0, 2.0)
    llm.script['late'] = reply(0.2, 0.0, 2.0)
    llm.gates['late'] = threading.Event()
    send(node, 'fwd')
    node.wait_for_workers(5)
    tick(node, clock)
    send(node, 'late')
    rec.clear()
    node.shutdown()
    assert rec.values() == [(0.0, 0.0)]
    llm.gates['late'].set()
    node.wait_for_workers(5)
    for _ in range(5):
        tick(node, clock)
    assert rec.values() == [(0.0, 0.0)]
