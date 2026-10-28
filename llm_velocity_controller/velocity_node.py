"""ROS 2 node: natural-language command -> LLM -> validated Twist on a test topic.

Threading model
- Subscription and timer callbacks run on the (single-threaded) executor.
- Each LLM request runs on its own daemon worker thread. Workers never touch
  the publisher; they put (generation, command, error) on a queue.
- The 10 Hz timer drains that queue and is the only place a moving Twist is
  published. Results whose generation is not the latest are discarded, so a
  slow, stale reply can never override a newer command.
- A new command and any STOP are handled in the subscription callback without
  waiting for inference: the active motion is cleared and zero is published.
- Progress is reported on STATUS_TOPIC as one JSON object per event (see
  STATUS_STATES), so a client such as `chat` can show what happened.
"""

import json
import queue
import signal
import threading
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from llm_velocity_controller.llm_client import DEFAULT_API_URL
from llm_velocity_controller.llm_client import DEFAULT_KEEP_ALIVE
from llm_velocity_controller.llm_client import DEFAULT_MODEL
from llm_velocity_controller.llm_client import DEFAULT_TIMEOUT_SEC
from llm_velocity_controller.llm_client import MockLLMClient
from llm_velocity_controller.llm_client import OllamaClient
from llm_velocity_controller.prompt import is_stop_command
from llm_velocity_controller.validator import validate

COMMAND_TOPIC = '/llm/user_command'
# Remap /cmd_vel to reach a robot under a namespace, e.g. /rubi_gazebo/cmd_vel.
OUTPUT_TOPIC = '/cmd_vel'
STATUS_TOPIC = '/llm/status'
PUBLISH_PERIOD_SEC = 0.1

# Every event published on STATUS_TOPIC carries one of these in "state".
STATUS_STATES = (
    'querying',   # command accepted, waiting for the LLM        (+ text)
    'stopped',    # STOP keyword, handled without the LLM        (+ text)
    'executing',  # validated, now publishing the velocity       (+ linear_x, angular_z, duration)
    'done',       # duration elapsed, zero published
    'llm_stop',   # LLM answered with an all-zero command
    'rejected',   # LLM failure or validation error, zero published (+ detail)
    'stale',      # late reply for an older command, ignored
    'shutdown',   # node stopping, zero published
)


def make_twist(linear_x=0.0, angular_z=0.0):
    twist = Twist()  # every other field stays 0.0
    twist.linear.x = float(linear_x)
    twist.angular.z = float(angular_z)
    return twist


class VelocityNode(Node):

    def __init__(self, llm_client=None, clock=time.monotonic):
        super().__init__('llm_velocity_controller')
        self.declare_parameter('llm_backend', 'ollama')
        self.declare_parameter('model', DEFAULT_MODEL)
        self.declare_parameter('api_url', DEFAULT_API_URL)
        self.declare_parameter('timeout_sec', DEFAULT_TIMEOUT_SEC)
        self.declare_parameter('keep_alive', DEFAULT_KEEP_ALIVE)
        self.declare_parameter('warm_up', True)
        self.declare_parameter('log_raw_response', True)
        self.declare_parameter('mock_delay_sec', 0.0)

        self._monotonic = clock
        self._lock = threading.Lock()
        self._generation = 0
        self._active = None          # (VelocityCommand, end_time, generation) or None
        self._results = queue.SimpleQueue()
        self._workers = []
        self._llm = llm_client or self._create_client()

        self._publisher = self.create_publisher(Twist, OUTPUT_TOPIC, 10)
        self._status_publisher = self.create_publisher(String, STATUS_TOPIC, 20)
        self._subscription = self.create_subscription(
            String, COMMAND_TOPIC, self._on_user_command, 10)
        self._timer = self.create_timer(PUBLISH_PERIOD_SEC, self._on_timer)
        self.get_logger().info(
            f'listening on {COMMAND_TOPIC}, publishing to {OUTPUT_TOPIC} at '
            f'{1.0 / PUBLISH_PERIOD_SEC:.0f} Hz using {type(self._llm).__name__}')
        if llm_client is None and self._param('warm_up') and hasattr(self._llm, 'warm_up'):
            threading.Thread(target=self._warm_up, name='llm-warm-up', daemon=True).start()

    def _warm_up(self):
        started = time.monotonic()
        try:
            self._llm.warm_up()
        except Exception as error:  # the node still works; first command is just slower
            self.get_logger().warn(f'LLM warm-up failed: {error}')
            return
        self.get_logger().info(f'LLM warm-up done in {time.monotonic() - started:.1f} s')

    def _param(self, name):
        return self.get_parameter(name).value

    def _create_client(self):
        raw_logger = None
        if self._param('log_raw_response'):
            def raw_logger(raw):
                self.get_logger().info(f'LLM raw response: {raw}')
        backend = self._param('llm_backend')
        if backend == 'mock':
            return MockLLMClient(self._param('mock_delay_sec'), raw_logger=raw_logger)
        if backend != 'ollama':
            raise ValueError(f"llm_backend must be 'ollama' or 'mock', got {backend!r}")
        return OllamaClient(model=self._param('model'), api_url=self._param('api_url'),
                            timeout_sec=float(self._param('timeout_sec')),
                            keep_alive=self._param('keep_alive'),
                            raw_logger=raw_logger)

    # -- publishing -----------------------------------------------------------

    def publish_zero(self):
        self._publisher.publish(make_twist())

    def _publish_status(self, generation, state, **fields):
        payload = {'id': generation, 'state': state, **fields}
        self._status_publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _cancel_active(self):
        """Invalidate in-flight requests and the running motion; returns new generation."""
        with self._lock:
            self._generation += 1
            self._active = None
            return self._generation

    # -- callbacks ------------------------------------------------------------

    def _on_user_command(self, msg):
        text = msg.data
        generation = self._cancel_active()
        self.publish_zero()
        if is_stop_command(text):
            self.get_logger().info(f'STOP: {text!r} (handled without LLM)')
            self._publish_status(generation, 'stopped', text=text)
            return
        self.get_logger().info(f'command #{generation}: {text!r} -> querying LLM')
        self._publish_status(generation, 'querying', text=text)
        worker = threading.Thread(
            target=self._run_llm, args=(generation, text),
            name=f'llm-request-{generation}', daemon=True)
        self._workers = [w for w in self._workers if w.is_alive()] + [worker]
        worker.start()

    def _run_llm(self, generation, text):
        try:
            command = validate(self._llm.complete(text))
            self._results.put((generation, command, None))
        except Exception as error:  # every failure must end in a zero command
            self._results.put((generation, None, error))

    def _on_timer(self):
        now = self._monotonic()
        self._apply_results(now)
        with self._lock:
            active = self._active
            if active is not None and now >= active[1]:
                self._active = None
        if active is None:
            return
        command, end_time, generation = active
        if now < end_time:
            self._publisher.publish(make_twist(command.linear_x, command.angular_z))
        else:
            self.publish_zero()
            self.get_logger().info('duration elapsed, published zero velocity')
            self._publish_status(generation, 'done')

    def _apply_results(self, now):
        while True:
            try:
                generation, command, error = self._results.get_nowait()
            except queue.Empty:
                return
            with self._lock:
                latest = self._generation
                stale = generation != latest
                started = (not stale and error is None and command.duration > 0.0
                           and not command.is_stop)
                if not stale:
                    self._active = ((command, now + command.duration, generation)
                                    if started else None)
            if stale:
                self.get_logger().warn(
                    f'discarded stale LLM result #{generation} (latest is #{latest})')
                self._publish_status(generation, 'stale')
            elif error is not None:
                self.publish_zero()
                self.get_logger().error(
                    f'command #{generation} rejected ({type(error).__name__}: {error}); '
                    'published zero velocity')
                self._publish_status(generation, 'rejected',
                                     detail=f'{type(error).__name__}: {error}')
            elif not started:
                self.publish_zero()
                self.get_logger().info(f'command #{generation}: LLM returned stop')
                self._publish_status(generation, 'llm_stop')
            else:
                self.get_logger().info(
                    f'command #{generation}: linear_x={command.linear_x} '
                    f'angular_z={command.angular_z} duration={command.duration}')
                self._publish_status(generation, 'executing', linear_x=command.linear_x,
                                     angular_z=command.angular_z, duration=command.duration)

    # -- lifecycle ------------------------------------------------------------

    def wait_for_workers(self, timeout_sec=None):
        for worker in list(self._workers):
            worker.join(timeout_sec)

    def shutdown(self):
        """Stop motion, ignore in-flight replies and publish zero."""
        generation = self._cancel_active()
        self.publish_zero()
        self._publish_status(generation, 'shutdown')


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # Keep the context alive on Ctrl+C so the final zero Twist can be published.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    # Explicit handlers: a process started in the background inherits SIGINT as
    # ignored, which would otherwise make Ctrl+C / kill -INT a no-op.
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    node = VelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.get_logger().info('shutting down, published zero velocity')
        time.sleep(0.1)  # let the middleware flush the final message
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
