"""Command the xArm 1S gripper by Cartesian pose in the base frame.

Frame: origin at the base center, URDF axes as-is -- -Y forward (base_yaw=0),
Z up, X left. Meters.

Each move is planned up front (all IK solved before the arm moves), then run
as a fixed-rate loop: a command per tick, then single-joint reads while they
fit. Closed loop then corrects until settled.
"""

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarm
from ikpy.chain import Chain

from . import config
from .kinematics import build_chain

# Retry seeds (shoulder, elbow, wrist deg) for when IK from the current pose
# lands in a local minimum; they span elbow-up and -down.
_FALLBACK_SEEDS_DEG = [(30, 80, -60), (-30, -80, 60), (60, 85, -75), (-60, -85, 75)]

# on_tick(angles_deg_by_name, gripper_open_fraction, remaining_ms): the tick's
# target, called last in the tick; must return within remaining_ms.
TickCallback = Callable[[dict, float, int], None]

# on_feedback(joint_name, sample): every joint read, failed ones included.
FeedbackCallback = Callable[[str, "FeedbackSample"], None]

# A read's slot in a tick: its reply, then the gap before the next command.
_READ_SLOT_S = (config.FEEDBACK_READ_BUDGET_MS + config.COMMAND_GAP_MS) / 1000

_ROLL = config.JOINTS[4]
_ROLL_INDEX = config.ACTIVE_LINK_INDICES[4]


@dataclass
class FeedbackSample:
    """One joint read (time.perf_counter() s): the measured angle (None if the
    read failed) and where the servo should have been then.
    """

    time_s: float
    measured_deg: float | None
    expected_deg: float


@dataclass
class MoveTiming:
    """A move's times (time.perf_counter() s): start, planned motion end
    (settling starts then), and return. `settled`: None open loop, else whether
    it settled (False = timed out). `corrections`: (time_s, joints nudged).
    """

    start_s: float
    motion_end_s: float
    end_s: float
    settled: bool | None
    corrections: list


class GripperPoseController:
    def __init__(
        self,
        com_port: str = "USB",
        dry_run: bool = False,
        rate_hz: float = 10.0,
        closed_loop: bool = False,
        kp: float = 0.5,
        settle_tolerance_deg: float = 2.0,
        settle_timeout_scale: float = 1.5,
        on_feedback: FeedbackCallback | None = None,
    ):
        """`rate_hz`: ticks per second; each fits ~3-4 reads at 10 Hz, 1 at 20 Hz,
        none above ~22 Hz (closed loop's limit).

        `closed_loop`: after each move, nudge commands by kp * (target - measured)
        until within `settle_tolerance_deg`; warn once the move has taken
        `settle_timeout_scale` x its duration (at least 3 more ticks).
        """
        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        if settle_timeout_scale < 1:
            raise ValueError(f"settle_timeout_scale must be >= 1, got {settle_timeout_scale}")
        if closed_loop and 1 / rate_hz < _READ_SLOT_S:
            raise ValueError(f"closed loop needs rate_hz <= {1 / _READ_SLOT_S:.1f} (room for one read per tick), got {rate_hz}")
        self.dry_run = dry_run
        self.period_s = 1 / rate_hz
        self.closed_loop = closed_loop
        self.kp = kp
        self.settle_tolerance_deg = settle_tolerance_deg
        self.settle_timeout_scale = settle_timeout_scale
        self.on_feedback = on_feedback
        self.last_move: MoveTiming | None = None

        self.chain = build_chain()
        # IK runs without wrist_roll: its axis runs through the tool point, so it
        # can't affect the solve. self.chain still does FK and commands.
        position_only_mask = list(self.chain.active_links_mask)
        position_only_mask[_ROLL_INDEX] = False
        self._position_chain = Chain(self.chain.links, active_links_mask=position_only_mask, name="xarm_1s_position_only")
        self._joint_angles_rad = np.zeros(len(self.chain.links))  # commanded
        self._gripper_open_fraction = 0.5
        self._command_ok_s = 0.0  # earliest time the next command may go out
        self._last_commands = None
        self._next_tick_s = 0.0
        self._late_ticks = 0
        self._next_read_index = 0
        # The last command: a straight joint-space line from _expected_from to
        # _expected_to over [_expected_start_s, _expected_end_s].
        self._expected_from = self._expected_to = self.get_joint_angles_deg()
        self._expected_start_s = self._expected_end_s = 0.0
        self._feedback: dict[str, FeedbackSample] = {}  # latest good read per joint
        # Closed loop: correction carried into later moves (e.g. gravity sag).
        self._settle_offset_deg = {joint.name: 0.0 for joint in config.JOINTS}
        self._corrections = []
        self._device = None if dry_run else xarm.Controller(com_port)

    def home(self, duration_ms: int = 1000, on_tick: TickCallback | None = None) -> None:
        """Move every joint to 0 deg and the gripper half open, from the measured
        pose (commanded where a read fails).
        """
        start_deg = {**self.get_joint_angles_deg(), **self._read_joint_angles_deg()}
        self._expected_from = self._expected_to = start_deg
        start = self._chain_angles_from_deg(start_deg)
        self._run_trajectory([start, np.zeros(len(self.chain.links))], 0.5, 0.5, duration_ms, on_tick)

    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        approach: tuple | None = None,
        roll_deg: float | None = None,
        gripper_open_fraction: float | None = None,
        duration_ms: int = 1000,
        linear: bool = True,
        position_tolerance_m: float | None = None,
        on_tick: TickCallback | None = None,
    ) -> np.ndarray:
        """Move the gripper tip to (x, y, z), pointing along `approach` (any
        length; None = free). `roll_deg` / `gripper_open_fraction` (None =
        unchanged) ramp alongside.

        `linear`: straight-line tip path (IK every config.LINEAR_MOVE_STEP_M);
        False moves in joint space. `position_tolerance_m` (default
        config.IK_POSITION_TOLERANCE_M) applies to the target only. All IK runs
        first, so an unreachable target raises ValueError before the arm moves.

        Returns the final chain angles (rad).
        """
        if position_tolerance_m is None:
            position_tolerance_m = config.IK_POSITION_TOLERANCE_M
        if gripper_open_fraction is None:
            gripper_open_fraction = self._gripper_open_fraction
        if roll_deg is not None:
            _ROLL.to_position(roll_deg)  # raises if outside its limits

        target_position = np.array([x, y, z], dtype=float)
        target_orientation = None
        if approach is not None:
            target_orientation = np.array(approach, dtype=float) / np.linalg.norm(approach)

        start_angles = self._joint_angles_rad.copy()
        start_position = self.get_pose()
        start_orientation = self.get_approach_vector()
        num_steps = max(1, math.ceil(np.linalg.norm(target_position - start_position) / config.LINEAR_MOVE_STEP_M)) if linear else 1

        keyframes = [start_angles]
        for step in range(1, num_steps + 1):
            fraction = step / num_steps
            step_position = start_position + fraction * (target_position - start_position)
            step_orientation = None
            if target_orientation is not None:
                step_orientation = start_orientation + fraction * (target_orientation - start_orientation)
                step_orientation /= np.linalg.norm(step_orientation)
            # The target gets the caller's tolerance and full effort; waypoints
            # are best-effort, capped, and biased toward the last one's branch.
            final = step == num_steps
            tolerance_m = position_tolerance_m if final else math.inf
            solution, error_m = self._solve_ik(
                keyframes[-1],
                step_position,
                step_orientation,
                tolerance_m,
                optimizer_budget=None if final else config.LINEAR_STEP_OPTIMIZER_BUDGET,
                regularization=None if final else config.LINEAR_STEP_REGULARIZATION,
            )
            if error_m > tolerance_m:
                raise ValueError(
                    f"Target pose ({x:.3f}, {y:.3f}, {z:.3f}) m is not reachable "
                    f"(residual error {error_m * 1000:.1f} mm, tolerance {position_tolerance_m * 1000:.1f} mm)."
                )
            keyframes.append(solution)

        if roll_deg is not None:
            start_roll, target_roll = start_angles[_ROLL_INDEX], math.radians(roll_deg)
            for i, keyframe in enumerate(keyframes):
                keyframe[_ROLL_INDEX] = start_roll + i / (len(keyframes) - 1) * (target_roll - start_roll)

        self._run_trajectory(keyframes, self._gripper_open_fraction, gripper_open_fraction, duration_ms, on_tick)
        return keyframes[-1]

    def move_joint(self, joint_name: str, angle_deg: float, duration_ms: int = 1000, on_tick: TickCallback | None = None) -> None:
        """Move one joint (a config.JOINTS name) directly, bypassing IK."""
        names = [joint.name for joint in config.JOINTS]
        if joint_name not in names:
            raise ValueError(f"Unknown joint {joint_name!r}. Known joints: {names}")
        config.JOINTS[names.index(joint_name)].to_position(angle_deg)  # raises if outside its limits
        target = self._joint_angles_rad.copy()
        target[config.ACTIVE_LINK_INDICES[names.index(joint_name)]] = math.radians(angle_deg)
        g = self._gripper_open_fraction
        self._run_trajectory([self._joint_angles_rad.copy(), target], g, g, duration_ms, on_tick)

    def set_gripper(self, open_fraction: float, duration_ms: int = 300, on_tick: TickCallback | None = None) -> None:
        hold = self._joint_angles_rad.copy()
        self._run_trajectory([hold, hold], self._gripper_open_fraction, max(0.0, min(1.0, open_fraction)), duration_ms, on_tick)

    def open_gripper(self, **kwargs) -> None:
        self.set_gripper(1.0, **kwargs)

    def close_gripper(self, **kwargs) -> None:
        self.set_gripper(0.0, **kwargs)

    def get_pose(self) -> np.ndarray:
        """The commanded tip position (x, y, z)."""
        return self.chain.forward_kinematics(self._joint_angles_rad)[:3, 3]

    def get_approach_vector(self) -> np.ndarray:
        """The commanded pointing direction (unit vector), as `approach` sets it."""
        return self.chain.forward_kinematics(self._joint_angles_rad)[:3, 2]

    def get_joint_angles_deg(self) -> dict:
        """Commanded angle of each arm joint, by name."""
        return self._angles_deg_by_name(self._joint_angles_rad)

    def get_measured_joint_angles_deg(self) -> dict:
        """Each arm joint's angle read from its servo (failed reads left out)."""
        return self._read_joint_angles_deg()

    # --- Execution ---

    def _run_trajectory(
        self, keyframes: list, gripper_start: float, gripper_end: float, duration_ms: int, on_tick: TickCallback | None
    ) -> None:
        """Run keyframes (chain angles, evenly timed over duration_ms) at the tick
        rate. Each tick commands the plan's pose for the next tick, then reads
        (so a slow read can't stall the arm). Returns once done (closed loop:
        settled), setting self.last_move.
        """
        duration_s = duration_ms / 1000
        num_ticks = max(1, math.ceil(duration_s / self.period_s - 1e-9))
        start_s = self._restart_ticks()
        for tick in range(num_ticks):
            self._wait_for_tick()
            target_s = min((tick + 1) * self.period_s, duration_s)
            fraction = target_s / duration_s if duration_s > 0 else 1.0
            angles_deg = self._angles_deg_by_name(self._sample_keyframes(keyframes, fraction))
            gripper = gripper_start + fraction * (gripper_end - gripper_start)
            self._command(self._with_settle_offset(angles_deg), gripper, arrive_s=start_s + target_s)
            self._read_feedback()
            self._call_on_tick(on_tick, angles_deg, gripper)

        self._joint_angles_rad = keyframes[-1].copy()
        self._gripper_open_fraction = gripper_end
        if self.dry_run:
            print(f"[dry run] {num_ticks} tick(s) @ {1 / self.period_s:g} Hz, final setPosition({self._last_commands})")
        self._corrections = []
        if self.closed_loop:
            settle_s = max(duration_s * self.settle_timeout_scale, duration_s + 3 * self.period_s)
            settled = self._settle(on_tick, deadline_s=start_s + settle_s)
            motion_end_s = min(start_s + duration_s, time.perf_counter())
        else:
            # Let the last command finish before the next move.
            remaining_s = start_s + duration_s - time.perf_counter()
            if remaining_s > 0:
                time.sleep(remaining_s)
            motion_end_s, settled = time.perf_counter(), None
        self.last_move = MoveTiming(start_s, motion_end_s, time.perf_counter(), settled, self._corrections)
        if self._late_ticks:
            print(f"    note: {self._late_ticks} tick(s) started late -- rate too high for this move's work?")

    def _settle(self, on_tick: TickCallback | None, deadline_s: float) -> bool:
        """Each tick, read every joint, then nudge each one that's off target by
        kp * (target - measured) -- but only once its tracking error has stopped
        changing, so a servo still catching up isn't corrected. Nudges accumulate
        (a steady sag goes to zero for 0 < kp < 2), are capped, and carry into
        later moves unless this times out at `deadline_s`. Returns whether it
        settled; nudges go to self._corrections.
        """
        target_deg = self.get_joint_angles_deg()
        offset_before = dict(self._settle_offset_deg)
        # A read counts once taken after its joint's last command arrived.
        valid_since = {name: self._expected_end_s for name in target_deg}
        previous_tracking = {}
        while True:
            tick_s = self._wait_for_tick(count_late=False)  # reading every joint may overrun a tick
            self._read_feedback(all_joints=True)
            valid = {name: s for name, s in self._feedback.items() if s.time_s >= valid_since[name]}
            residual = {name: target_deg[name] - s.measured_deg for name, s in valid.items()}
            if len(residual) == len(target_deg) and max(map(abs, residual.values())) <= self.settle_tolerance_deg:
                return True
            if tick_s >= deadline_s:
                print(
                    "    warning: settle timed out, residual "
                    + ", ".join(f"{name}={residual[name]:+.1f}" if name in residual else f"{name}=?" for name in target_deg)
                    + " deg"
                )
                self._settle_offset_deg = offset_before
                return False
            nudged = []
            for name, sample in valid.items():
                if sample.time_s < tick_s:  # judged on an earlier tick
                    continue
                tracking = sample.expected_deg - sample.measured_deg
                steady = name in previous_tracking and abs(tracking - previous_tracking[name]) <= config.SETTLE_STEADY_DEG
                previous_tracking[name] = tracking
                if steady and abs(residual[name]) > self.settle_tolerance_deg:
                    limit = config.SETTLE_MAX_CORRECTION_DEG
                    self._settle_offset_deg[name] = float(np.clip(self._settle_offset_deg[name] + self.kp * residual[name], -limit, limit))
                    nudged.append(name)
            if nudged:
                now_s = time.perf_counter()
                self._corrections.append((now_s, nudged))
                # Small, so it finishes well before the next tick's read.
                arrive_s = now_s + min(self.period_s, config.SETTLE_NUDGE_S)
                self._command(self._with_settle_offset(target_deg), self._gripper_open_fraction, arrive_s=arrive_s)
                valid_since.update({name: self._expected_end_s for name in nudged})
            self._call_on_tick(on_tick, target_deg, self._gripper_open_fraction)

    def _with_settle_offset(self, angles_deg_by_name: dict) -> dict:
        return {
            joint.name: float(np.clip(angles_deg_by_name[joint.name] + self._settle_offset_deg[joint.name], *joint.limit_deg))
            for joint in config.JOINTS
        }

    @staticmethod
    def _sample_keyframes(keyframes: list, fraction: float) -> np.ndarray:
        """Angles `fraction` (0-1) of the way along evenly timed keyframes."""
        position = fraction * (len(keyframes) - 1)
        index = min(int(position), len(keyframes) - 2)
        return keyframes[index] + (position - index) * (keyframes[index + 1] - keyframes[index])

    def _command(self, angles_deg_by_name: dict, gripper_open_fraction: float, arrive_s: float) -> None:
        """Send one command for the servos to finish at `arrive_s`."""
        self._wait_for_command_gap()
        now_s = time.perf_counter()
        self._expected_from = self._expected_at(now_s)
        self._expected_to = dict(angles_deg_by_name)
        self._expected_start_s, self._expected_end_s = now_s, max(now_s, arrive_s)
        self._send(self._commands_for(angles_deg_by_name, gripper_open_fraction), max(1, round((arrive_s - now_s) * 1000)))

    def _expected_at(self, time_s: float) -> dict:
        """Where the servos should be at `time_s`, per the last command."""
        span_s = self._expected_end_s - self._expected_start_s
        t = 1.0 if span_s <= 0 else min(1.0, max(0.0, (time_s - self._expected_start_s) / span_s))
        return {name: self._expected_from[name] + t * (self._expected_to[name] - self._expected_from[name]) for name in self._expected_to}

    # --- Tick timing ---

    def _restart_ticks(self) -> float:
        """Start a move's tick clock now (time since the last move, e.g. IK, isn't late)."""
        self._next_tick_s = time.perf_counter()
        self._late_ticks = 0
        return self._next_tick_s

    def _wait_for_tick(self, count_late: bool = True) -> float:
        """Sleep until the next tick; return its scheduled time. A late tick
        restarts the clock from now (no burst to catch up).
        """
        now_s = time.perf_counter()
        if now_s < self._next_tick_s:
            time.sleep(self._next_tick_s - now_s)
        elif now_s - self._next_tick_s > config.TICK_LATE_SLACK_S:
            self._late_ticks += count_late
            self._next_tick_s = now_s
        tick_s = self._next_tick_s
        self._next_tick_s += self.period_s
        return tick_s

    def _call_on_tick(self, on_tick: TickCallback | None, angles_deg_by_name: dict, gripper_open_fraction: float) -> None:
        if on_tick is not None:
            on_tick(angles_deg_by_name, gripper_open_fraction, max(0, int((self._next_tick_s - time.perf_counter()) * 1000)))

    # --- Position feedback ---

    def _read_feedback(self, all_joints: bool = False) -> None:
        """Read joints one request each (all-at-once reads fail often and make
        the servos hitch), rotating, while a read still fits before the next
        tick (or all of them). Passes every read to on_feedback.
        """
        for _ in config.JOINTS:
            if not all_joints and time.perf_counter() + _READ_SLOT_S > self._next_tick_s:
                break
            joint = config.JOINTS[self._next_read_index]
            self._next_read_index = (self._next_read_index + 1) % len(config.JOINTS)
            measured_deg = self._read_joint_angles_deg([joint]).get(joint.name)
            time_s = time.perf_counter()
            sample = FeedbackSample(time_s, measured_deg, self._expected_at(time_s)[joint.name])
            if measured_deg is not None:
                self._feedback[joint.name] = sample
            if self.on_feedback is not None:
                self.on_feedback(joint.name, sample)

    def _read_joint_angles_deg(self, joints: list | None = None) -> dict:
        """Angles of `joints` (default all), one request each; failed reads left
        out. Dry run returns where the servos should be.
        """
        joints = config.JOINTS if joints is None else joints
        if self.dry_run:
            expected = self._expected_at(time.perf_counter())
            return {joint.name: expected[joint.name] for joint in joints}
        measured = {}
        for joint in joints:
            position = self._read_servo_positions([joint.servo_id]).get(joint.servo_id)
            if position is not None:
                measured[joint.name] = joint.to_angle_deg(position)
        return measured

    def _read_servo_positions(self, servo_ids: list) -> dict:
        """Raw positions by servo ID from one request (non-answering servos left
        out). Unlike xarm's getPosition, drains stale replies first and validates
        the reply -- though a late reply to an identical request can't be told
        apart, so after a timeout the next read may be a read-period old.
        """
        device = self._device
        if device._is_serial:  # only USB HID is reimplemented
            servos = [xarm.Servo(servo_id) for servo_id in servo_ids]
            try:
                device.getPosition(servos)
                return {servo.servo_id: servo.position for servo in servos}
            except Exception:
                return {}
            finally:
                self._start_command_gap()
        hid_device = device._device
        while hid_device.read(64, 0):  # a timed-out read's late reply
            pass
        device._send(device.CMD_GET_SERVO_POSITION, bytearray([len(servo_ids), *servo_ids]))
        deadline_s = time.perf_counter() + config.FEEDBACK_READ_TIMEOUT_MS / 1000
        try:
            while (remaining_ms := int((deadline_s - time.perf_counter()) * 1000)) > 0:
                report = hid_device.read(64, remaining_ms)
                # 0x55 0x55 len cmd count, then (id pos_lo pos_hi) per servo
                if len(report) >= 5 and report[0] == report[1] == device.SIGNATURE and report[3] == device.CMD_GET_SERVO_POSITION:
                    positions = {}
                    for i in range(report[4]):
                        entry = report[5 + 3 * i : 8 + 3 * i]
                        if len(entry) == 3 and entry[0] in servo_ids:
                            position = entry[2] * 256 + entry[1]
                            if config.SERVO_POSITION_MIN <= position <= config.SERVO_POSITION_MAX:
                                positions[entry[0]] = position
                    return positions
            return {}
        finally:
            self._start_command_gap()

    def _wait_for_command_gap(self) -> None:
        remaining_s = self._command_ok_s - time.perf_counter()
        if remaining_s > 0:
            time.sleep(remaining_s)

    def _start_command_gap(self) -> None:
        self._command_ok_s = time.perf_counter() + config.COMMAND_GAP_MS / 1000

    # --- Angle/command conversion ---

    @staticmethod
    def _angles_deg_by_name(chain_angles_rad: np.ndarray) -> dict:
        return {joint.name: math.degrees(chain_angles_rad[i]) for i, joint in zip(config.ACTIVE_LINK_INDICES, config.JOINTS)}

    def _chain_angles_from_deg(self, angles_deg_by_name: dict) -> np.ndarray:
        angles_rad = np.zeros(len(self.chain.links))
        for i, joint in zip(config.ACTIVE_LINK_INDICES, config.JOINTS):
            angles_rad[i] = math.radians(angles_deg_by_name[joint.name])
        return angles_rad

    @staticmethod
    def _commands_for(angles_deg_by_name: dict, gripper_open_fraction: float) -> list:
        gripper = config.GRIPPER_CLOSED_POS + gripper_open_fraction * (config.GRIPPER_OPEN_POS - config.GRIPPER_CLOSED_POS)
        return [(joint.servo_id, joint.to_position(angles_deg_by_name[joint.name])) for joint in config.JOINTS] + [
            (config.GRIPPER_SERVO_ID, int(round(gripper)))
        ]

    def _send(self, commands: list, duration_ms: int) -> None:
        self._last_commands = commands
        if self.dry_run:
            return
        self._device.setPosition(commands, duration=duration_ms, wait=False)
        self._start_command_gap()

    # --- IK ---

    def _solve_ik(
        self,
        seed_rad: np.ndarray,
        target_position: np.ndarray,
        target_orientation: np.ndarray | None,
        tolerance_m: float,
        optimizer_budget: int | None = None,
        regularization: float | None = None,
    ) -> tuple:
        """IK from `seed_rad`, retrying from fallback seeds until one is within
        the stricter of config's tolerance and `tolerance_m`. `optimizer_budget`
        caps scipy evals per seed; `regularization` (ikpy's) biases toward the
        seed's branch. Returns (best solution, its error in m).
        """
        best_solution, best_error = None, np.inf
        good_enough_m = min(config.IK_POSITION_TOLERANCE_M, tolerance_m)
        for seed in self._ik_seeds(seed_rad):
            solution = self._position_chain.inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                orientation_mode="Z" if target_orientation is not None else None,
                initial_position=seed,
                optimizer_budget=optimizer_budget,
                regularization_parameter=regularization,
            )
            error_m = float(np.linalg.norm(self.chain.forward_kinematics(solution)[:3, 3] - target_position))
            if error_m < best_error:
                best_solution, best_error = solution, error_m
            if best_error <= good_enough_m:
                break
        return best_solution, best_error

    @staticmethod
    def _ik_seeds(seed_rad: np.ndarray):
        """`seed_rad`, then copies with the pitch joints set to each fallback
        (roll kept, as the IK chain doesn't move it).
        """
        yield seed_rad
        pitch_joints = list(zip(config.ACTIVE_LINK_INDICES[1:4], config.JOINTS[1:4]))
        for fallback_deg in _FALLBACK_SEEDS_DEG:
            seed = seed_rad.copy()
            for (i, joint), angle_deg in zip(pitch_joints, fallback_deg):
                seed[i] = math.radians(np.clip(angle_deg, *joint.limit_deg))
            yield seed
