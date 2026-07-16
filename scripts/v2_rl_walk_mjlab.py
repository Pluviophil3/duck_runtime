import argparse
import atexit
from datetime import datetime
import os
import sys
import time

import numpy as np

from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.rl_utils import LowPassActionFilter, make_action_dict
from mini_bdx_runtime.rustypot_position_hwi import HWI


HOME_DIR = os.path.expanduser("~")

JOINT_ORDER_16 = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "left_antenna",
    "right_antenna",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

ACTION_ORDER_14 = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

ACTION_SCALE_BY_JOINT = {
    "left_hip_yaw": 0.13,
    "left_hip_roll": 0.13,
    "left_hip_pitch": 0.13,
    "left_knee": 0.13,
    "left_ankle": 0.13,
    "right_hip_yaw": 0.13,
    "right_hip_roll": 0.13,
    "right_hip_pitch": 0.13,
    "right_knee": 0.13,
    "right_ankle": 0.13,
    "neck_pitch": 0.10,
    "head_pitch": 0.10,
    "head_yaw": 0.10,
    "head_roll": 0.10,
}

ANCHOR_BODY_INDEX = 0
OBS_DIM = 87
ACTION_DIM = 14

JOINT_LIMITS_BY_JOINT = {
    "left_hip_yaw": (-0.39, 0.33),
    "left_hip_roll": (-0.46, 0.53),
    "left_hip_pitch": (-0.45, 0.17),
    "left_knee": (-0.38, 0.77),
    "left_ankle": (-0.756, 0.704),
    "neck_pitch": (0.0, 0.7),
    "head_pitch": (-0.7, 0.54),
    "head_yaw": (-0.45, 0.58),
    "head_roll": (-0.6, 0.7),
    "right_hip_yaw": (-0.369, 0.502),
    "right_hip_roll": (-0.584, 0.42),
    "right_hip_pitch": (-0.23, 0.47),
    "right_knee": (-0.373, 0.753),
    "right_ankle": (-0.765, 0.676),
}

OBS_SLICES = (
    ("ref_pos", 0, 16),
    ("ref_vel", 16, 32),
    ("anchor_ori", 32, 38),
    ("gyro", 38, 41),
    ("joint_pos", 41, 57),
    ("joint_vel", 57, 73),
    ("last_action", 73, 87),
)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_log_file(log_path):
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join("logs", f"mjlab_runtime_{timestamp}.log")
    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    def close_log():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(close_log)
    print(f"[mjlab] logging to {log_path}")
    return log_path


def quat_normalize_wxyz(q):
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


def quat_inv_wxyz(q):
    q = quat_normalize_wxyz(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_to_matrix_wxyz(q):
    w, x, y, z = quat_normalize_wxyz(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def motion_anchor_ori_b(robot_quat_wxyz, motion_anchor_quat_wxyz):
    rel = quat_mul_wxyz(quat_inv_wxyz(robot_quat_wxyz), motion_anchor_quat_wxyz)
    mat = quat_to_matrix_wxyz(rel)
    return mat[:, :2].reshape(-1).astype(np.float32)


class MjlabMotionReference:
    def __init__(self, motion_path):
        data = np.load(motion_path, allow_pickle=True)
        raw_joint_pos = data["joint_pos"].astype(np.float32)
        raw_joint_vel = data["joint_vel"].astype(np.float32)
        self.source_joint_names = self._source_joint_names(data, raw_joint_pos)
        self.joint_pos, self.joint_vel = self._to_joint_order_16(
            raw_joint_pos,
            raw_joint_vel,
            self.source_joint_names,
        )
        self.body_quat_w = data["body_quat_w"].astype(np.float32)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])

        if self.body_quat_w.shape[1] <= ANCHOR_BODY_INDEX:
            raise ValueError("motion body_quat_w does not contain the anchor body")

    @staticmethod
    def _source_joint_names(data, joint_pos):
        if "joint_names" in data.files:
            names = tuple(str(name) for name in data["joint_names"])
            if len(names) != joint_pos.shape[1]:
                raise ValueError(
                    "motion joint_names length "
                    f"{len(names)} != joint_pos dim {joint_pos.shape[1]}"
                )
            return names
        if joint_pos.shape[1] == len(JOINT_ORDER_16):
            return JOINT_ORDER_16
        raise ValueError(
            "motion is missing joint_names; cannot map "
            f"{joint_pos.shape[1]} joint columns to runtime order"
        )

    @staticmethod
    def _to_joint_order_16(joint_pos, joint_vel, source_joint_names):
        if joint_pos.shape != joint_vel.shape:
            raise ValueError(
                f"motion joint_pos shape {joint_pos.shape} != joint_vel shape {joint_vel.shape}"
            )
        source_index = {name: index for index, name in enumerate(source_joint_names)}
        aligned_pos = np.zeros((joint_pos.shape[0], len(JOINT_ORDER_16)), dtype=np.float32)
        aligned_vel = np.zeros_like(aligned_pos)
        missing = []
        for target_index, target_name in enumerate(JOINT_ORDER_16):
            source_i = source_index.get(target_name)
            if source_i is None:
                missing.append(target_name)
                continue
            aligned_pos[:, target_index] = joint_pos[:, source_i]
            aligned_vel[:, target_index] = joint_vel[:, source_i]

            backlash_i = source_index.get(f"{target_name}_backlash")
            if backlash_i is not None:
                aligned_pos[:, target_index] += joint_pos[:, backlash_i]
                aligned_vel[:, target_index] += joint_vel[:, backlash_i]

        if missing:
            raise ValueError(
                "motion is missing runtime joints required by JOINT_ORDER_16: "
                f"{missing}; source joints={source_joint_names}"
            )
        return aligned_pos, aligned_vel

    @property
    def num_frames(self):
        return self.joint_pos.shape[0]

    def frame(self, index):
        i = index % self.num_frames
        return (
            self.joint_pos[i],
            self.joint_vel[i],
            self.body_quat_w[i, ANCHOR_BODY_INDEX],
        )


class MjlabRLWalk:
    def __init__(
        self,
        onnx_model_path,
        motion_path,
        duck_config_path=f"{HOME_DIR}/duck_config.json",
        serial_port="/dev/ttyACM0",
        control_freq=50,
        pid=(30, 0, 0),
        pitch_bias=0.0,
        cutoff_frequency=None,
        max_target_step=0.08,
        action_gain=1.0,
        initial_pose="motion_start",
        start_frame=0,
        action_clip=None,
        joint_vel_scale=1.0,
        safety_clip=True,
        joint_limit_margin=0.02,
        debug_interval_s=1.0,
        debug_top_k=5,
        dry_run=False,
        debug=False,
    ):
        self.debug = debug
        self.dry_run = dry_run
        self.control_freq = control_freq
        self.max_target_step = max_target_step
        self.action_gain = action_gain
        self.initial_pose = initial_pose
        self.start_frame = start_frame
        self.action_clip = action_clip
        self.joint_vel_scale = joint_vel_scale
        self.safety_clip = safety_clip
        self.joint_limit_margin = joint_limit_margin
        self.debug_interval_steps = max(1, int(round(debug_interval_s * control_freq)))
        self.debug_top_k = debug_top_k

        self.policy = OnnxInfer(onnx_model_path, awd=True)
        self.motion = MjlabMotionReference(motion_path)
        self.duck_config = DuckConfig(duck_config_path)
        self.imu = Imu(
            sampling_freq=int(control_freq),
            user_pitch_bias=pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )
        self.hwi = None if dry_run else HWI(self.duck_config, serial_port)
        self.pid = pid
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.motion_i = start_frame
        self.last_timing = {}
        self.last_raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_target_scaled = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_target_rate_limited = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_target_safety_input = np.zeros(ACTION_DIM, dtype=np.float32)

        self.action_scale = np.array(
            [ACTION_SCALE_BY_JOINT[name] for name in ACTION_ORDER_14],
            dtype=np.float32,
        )
        self.joint_lower = np.array(
            [JOINT_LIMITS_BY_JOINT[name][0] for name in ACTION_ORDER_14],
            dtype=np.float32,
        )
        self.joint_upper = np.array(
            [JOINT_LIMITS_BY_JOINT[name][1] for name in ACTION_ORDER_14],
            dtype=np.float32,
        )
        self.clip_lower = self.joint_lower + self.joint_limit_margin
        self.clip_upper = self.joint_upper - self.joint_limit_margin
        self.motor_targets = np.zeros(ACTION_DIM, dtype=np.float32)
        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(control_freq, cutoff_frequency)

        self._check_runtime_order()
        self._print_startup_summary(onnx_model_path, motion_path)
        if not dry_run:
            self.start(self._initial_pose_targets())
            current_pos = self.hwi.get_present_positions()
            if current_pos is not None and len(current_pos) == ACTION_DIM:
                self.motor_targets = current_pos.astype(np.float32)
                if self.action_filter is not None:
                    self.action_filter.last_action = self.motor_targets.copy()
                    self.action_filter.current_action = self.motor_targets.copy()
                self._warn_limit_violations("current_after_start", self.motor_targets)

    def _print_startup_summary(self, onnx_model_path, motion_path):
        if not self.debug:
            return
        print("[mjlab] startup")
        print(f"  policy: {onnx_model_path}")
        print(f"  motion: {motion_path}")
        print(f"  control_freq: {self.control_freq} Hz")
        print(f"  start_frame: {self.start_frame % self.motion.num_frames}")
        print(f"  action_gain: {self.action_gain}")
        print(f"  action_clip: {self.action_clip}")
        print(f"  joint_vel_scale: {self.joint_vel_scale}")
        print(f"  max_target_step: {self.max_target_step}")
        print(f"  safety_clip: {self.safety_clip}, margin: {self.joint_limit_margin}")
        print("  action order / scale / logical limits:")
        for name, scale, lo, hi in zip(
            ACTION_ORDER_14, self.action_scale, self.joint_lower, self.joint_upper
        ):
            print(f"    {name:16s} scale={scale:.3f} limit=[{lo:.3f}, {hi:.3f}]")

    def _check_runtime_order(self):
        if len(ACTION_ORDER_14) != ACTION_DIM:
            raise ValueError("ACTION_ORDER_14 length mismatch")
        if len(JOINT_ORDER_16) != 16:
            raise ValueError("JOINT_ORDER_16 length mismatch")
        if tuple(j for j in JOINT_ORDER_16 if "antenna" not in j) != ACTION_ORDER_14:
            raise ValueError("Action order must be joint order with antennas removed")
        if self.hwi is not None and tuple(self.hwi.joints.keys()) != ACTION_ORDER_14:
            raise ValueError(
                "HWI joint order does not match mjlab action order:\n"
                f"HWI:    {tuple(self.hwi.joints.keys())}\n"
                f"mjlab:  {ACTION_ORDER_14}"
            )

    def _initial_pose_targets(self):
        if self.initial_pose == "zero":
            values = np.zeros(ACTION_DIM, dtype=np.float32)
        elif self.initial_pose == "motion_start":
            joint_pos_16, _, _ = self.motion.frame(self.motion_i)
            values_by_name = dict(zip(JOINT_ORDER_16, joint_pos_16))
            values = np.array(
                [values_by_name[name] for name in ACTION_ORDER_14],
                dtype=np.float32,
            )
        else:
            raise ValueError(
                f"Unsupported initial_pose {self.initial_pose!r}; "
                "use 'motion_start' or 'zero'"
            )
        return dict(zip(ACTION_ORDER_14, values))

    def start(self, target_pos):
        kps = [self.pid[0]] * ACTION_DIM
        kds = [self.pid[2]] * ACTION_DIM
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on(target_pos=target_pos)
        time.sleep(1.0)

    def _read_joint_state_16(self):
        pos_14 = np.zeros(ACTION_DIM, dtype=np.float32)
        vel_14 = np.zeros(ACTION_DIM, dtype=np.float32)
        if self.hwi is not None:
            pos_14 = self.hwi.get_present_positions()
            vel_14 = self.hwi.get_present_velocities()
            if pos_14 is None or vel_14 is None:
                return None, None
            pos_14 = pos_14.astype(np.float32)
            vel_14 = vel_14.astype(np.float32) * self.joint_vel_scale

        pos_by_name = dict(zip(ACTION_ORDER_14, pos_14))
        vel_by_name = dict(zip(ACTION_ORDER_14, vel_14))
        pos_16 = np.array(
            [pos_by_name.get(name, 0.0) for name in JOINT_ORDER_16],
            dtype=np.float32,
        )
        vel_16 = np.array(
            [vel_by_name.get(name, 0.0) for name in JOINT_ORDER_16],
            dtype=np.float32,
        )
        return pos_16, vel_16

    def get_obs(self):
        ref_pos, ref_vel, ref_anchor_quat = self.motion.frame(self.motion_i)
        imu_data = self.imu.get_data()
        joint_pos, joint_vel = self._read_joint_state_16()
        if joint_pos is None or joint_vel is None:
            return None

        robot_quat = np.asarray(imu_data["quat_wxyz"], dtype=np.float32)
        anchor_ori = motion_anchor_ori_b(robot_quat, ref_anchor_quat)
        gyro = np.asarray(imu_data["gyro"], dtype=np.float32)

        obs = np.concatenate(
            [
                ref_pos,
                ref_vel,
                anchor_ori,
                gyro,
                joint_pos,
                joint_vel,
                self.last_action,
            ]
        ).astype(np.float32)
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"obs shape {obs.shape} != ({OBS_DIM},)")
        return obs

    def _limit_margin(self, values):
        return np.minimum(values - self.joint_lower, self.joint_upper - values)

    def _warn_limit_violations(self, label, values):
        low = values < self.joint_lower
        high = values > self.joint_upper
        near = self._limit_margin(values) < self.joint_limit_margin
        mask = low | high | near
        if not np.any(mask):
            return
        print(f"[mjlab][limit] {label}")
        for i in np.where(mask)[0]:
            name = ACTION_ORDER_14[i]
            state = "LOW" if low[i] else "HIGH" if high[i] else "NEAR"
            print(
                f"  {state:4s} {name:16s} value={values[i]: .4f} "
                f"limit=[{self.joint_lower[i]: .4f}, {self.joint_upper[i]: .4f}]"
            )

    def _format_top_abs(self, label, values, names, top_k=None):
        top_k = self.debug_top_k if top_k is None else top_k
        order = np.argsort(-np.abs(values))[:top_k]
        parts = [f"{names[i]}={values[i]:+.3f}" for i in order]
        return f"{label}: " + ", ".join(parts)

    def _process_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape {action.shape} != ({ACTION_DIM},)")
        if self.action_clip is None:
            return action
        clipped = np.clip(action, -self.action_clip, self.action_clip)
        if self.debug and np.any(np.abs(clipped - action) > 1e-6):
            print(f"[mjlab][action_clip] frame={self.motion_i % self.motion.num_frames}")
            for i in np.where(np.abs(clipped - action) > 1e-6)[0]:
                print(
                    f"  {ACTION_ORDER_14[i]:16s} raw={action[i]: .4f} "
                    f"clipped={clipped[i]: .4f}"
                )
        return clipped.astype(np.float32)

    def _targets_from_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape {action.shape} != ({ACTION_DIM},)")
        target = action * self.action_scale * self.action_gain
        self.last_target_scaled = target.astype(np.float32)
        if self.max_target_step is not None and self.motion_i > 0:
            delta = np.clip(
                target - self.motor_targets,
                -self.max_target_step,
                self.max_target_step,
            )
            target = self.motor_targets + delta
        self.last_target_rate_limited = target.astype(np.float32)
        unclipped = target.astype(np.float32)
        self.last_target_safety_input = unclipped.copy()
        if self.safety_clip:
            target = np.clip(unclipped, self.clip_lower, self.clip_upper)
            clipped = np.abs(target - unclipped) > 1e-6
            if np.any(clipped):
                print(f"[mjlab][clip] frame={self.motion_i % self.motion.num_frames}")
                for i in np.where(clipped)[0]:
                    print(
                        f"  {ACTION_ORDER_14[i]:16s} raw={unclipped[i]: .4f} "
                        f"clipped={target[i]: .4f} "
                        f"safe=[{self.clip_lower[i]: .4f}, {self.clip_upper[i]: .4f}]"
                    )
        else:
            self._warn_limit_violations("target_unclipped", unclipped)
        return target.astype(np.float32)

    def _debug_print_step(self, obs, action, target):
        ref_pos = obs[0:16]
        ref_vel = obs[16:32]
        joint_pos_16 = obs[41:57]
        joint_vel_16 = obs[57:73]
        pos_by_name = dict(zip(JOINT_ORDER_16, joint_pos_16))
        vel_by_name = dict(zip(JOINT_ORDER_16, joint_vel_16))
        ref_by_name = dict(zip(JOINT_ORDER_16, ref_pos))
        current_14 = np.array(
            [pos_by_name[name] for name in ACTION_ORDER_14],
            dtype=np.float32,
        )
        vel_14 = np.array(
            [vel_by_name[name] for name in ACTION_ORDER_14],
            dtype=np.float32,
        )
        ref_14 = np.array(
            [ref_by_name[name] for name in ACTION_ORDER_14],
            dtype=np.float32,
        )
        target_error = target - current_14
        ref_error = current_14 - ref_14
        obs_norms = ", ".join(
            f"{name}={np.linalg.norm(obs[start:end]):.3f}"
            for name, start, end in OBS_SLICES
        )
        print(
            "[mjlab] "
            f"frame={self.motion_i % self.motion.num_frames} "
            f"obs_norm={np.linalg.norm(obs):.3f} "
            f"action_abs_max={np.max(np.abs(action)):.3f} "
            f"target_abs_max={np.max(np.abs(target)):.3f} "
            f"current_abs_max={np.max(np.abs(current_14)):.3f}"
        )
        print(f"  obs: {obs_norms}")
        if self.last_timing:
            step_ms = self.last_timing["step_ms"]
            compute_hz = 1000.0 / step_ms if step_ms > 1e-6 else float("inf")
            budget_ms = 1000.0 / self.control_freq
            print(
                "  timing: "
                f"obs={self.last_timing['obs_ms']:.2f}ms "
                f"infer={self.last_timing['infer_ms']:.2f}ms "
                f"target={self.last_timing['target_ms']:.2f}ms "
                f"command={self.last_timing['command_ms']:.2f}ms "
                f"step={step_ms:.2f}ms "
                f"compute_hz={compute_hz:.1f} "
                f"budget={budget_ms:.2f}ms"
            )
        print("  " + self._format_top_abs("raw_action", self.last_raw_action, ACTION_ORDER_14))
        print("  " + self._format_top_abs("action", action, ACTION_ORDER_14))
        print(
            "  "
            + self._format_top_abs(
                "scaled_target", self.last_target_scaled, ACTION_ORDER_14
            )
        )
        print(
            "  "
            + self._format_top_abs(
                "rate_limit_delta",
                self.last_target_rate_limited - self.last_target_scaled,
                ACTION_ORDER_14,
            )
        )
        print(
            "  "
            + self._format_top_abs(
                "safety_clip_delta",
                target - self.last_target_safety_input,
                ACTION_ORDER_14,
            )
        )
        print("  " + self._format_top_abs("target-current", target_error, ACTION_ORDER_14))
        print("  " + self._format_top_abs("current-ref", ref_error, ACTION_ORDER_14))
        print("  " + self._format_top_abs("joint_vel", vel_14, ACTION_ORDER_14))
        self._warn_limit_violations("current", current_14)
        self._warn_limit_violations("target", target)

    def step(self):
        step_t0 = time.perf_counter()
        obs_t0 = time.perf_counter()
        obs = self.get_obs()
        if obs is None:
            return False
        obs_t1 = time.perf_counter()
        infer_t0 = time.perf_counter()
        raw_action = self.policy.infer(obs).astype(np.float32)
        self.last_raw_action = raw_action.copy()
        infer_t1 = time.perf_counter()
        action_t0 = time.perf_counter()
        action = self._process_action(raw_action)
        target = self._targets_from_action(action)

        if self.action_filter is not None:
            self.action_filter.push(target)
            target = self.action_filter.get_filtered_action()
        action_t1 = time.perf_counter()

        self.last_action = action.copy()
        self.motor_targets = target.copy()

        command_t0 = time.perf_counter()
        if self.hwi is not None:
            self.hwi.set_position_all(make_action_dict(target, ACTION_ORDER_14))
        command_t1 = time.perf_counter()

        step_t1 = time.perf_counter()
        self.last_timing = {
            "obs_ms": (obs_t1 - obs_t0) * 1000.0,
            "infer_ms": (infer_t1 - infer_t0) * 1000.0,
            "target_ms": (action_t1 - action_t0) * 1000.0,
            "command_ms": (command_t1 - command_t0) * 1000.0,
            "step_ms": (step_t1 - step_t0) * 1000.0,
        }

        if self.debug and self.motion_i % self.debug_interval_steps == 0:
            self._debug_print_step(obs, action, target)

        self.motion_i += 1
        return True

    def run(self):
        print("Starting mjlab tracking runtime")
        print(f"motion frames: {self.motion.num_frames}, fps: {self.motion.fps}")
        try:
            while True:
                t = time.time()
                self.step()
                took = time.time() - t
                time.sleep(max(0.0, 1.0 / self.control_freq - took))
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            if self.hwi is not None:
                self.hwi.turn_off()
            print("Stopped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", required=True)
    parser.add_argument("--motion_path", required=True)
    parser.add_argument("--duck_config_path", default=f"{HOME_DIR}/duck_config.json")
    parser.add_argument("--serial_port", default="/dev/ttyACM0")
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0.0)
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument("--max_target_step", type=float, default=0.08)
    parser.add_argument("--action_gain", type=float, default=1.0)
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Reference motion frame used for startup and initial pose.",
    )
    parser.add_argument(
        "--action_clip",
        type=float,
        default=None,
        help="Optional symmetric clip applied to raw policy actions before targets.",
    )
    parser.add_argument(
        "--joint_vel_scale",
        type=float,
        default=1.0,
        help="Scale applied to hardware joint velocities before building observations.",
    )
    parser.add_argument(
        "--no_safety_clip",
        action="store_true",
        help="Disable MJCF joint-range clipping for target positions.",
    )
    parser.add_argument(
        "--joint_limit_margin",
        type=float,
        default=0.02,
        help="Safety margin in radians inside each MJCF joint range.",
    )
    parser.add_argument(
        "--debug_interval_s",
        type=float,
        default=1.0,
        help="Seconds between detailed debug prints when --debug is enabled.",
    )
    parser.add_argument(
        "--debug_top_k",
        type=int,
        default=5,
        help="Number of largest per-joint values to print in debug summaries.",
    )
    parser.add_argument(
        "--initial_pose",
        choices=("motion_start", "zero"),
        default="motion_start",
        help="Joint target used during motor turn-on before policy starts.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--log_path",
        default=None,
        help=(
            "Write terminal/debug output to this log file as well. "
            "When omitted with --debug, a timestamped file is created under logs/."
        ),
    )
    args = parser.parse_args()

    if args.debug or args.log_path is not None:
        setup_log_file(args.log_path)

    runner = MjlabRLWalk(
        onnx_model_path=args.onnx_model_path,
        motion_path=args.motion_path,
        duck_config_path=args.duck_config_path,
        serial_port=args.serial_port,
        control_freq=args.control_freq,
        pid=(args.p, args.i, args.d),
        pitch_bias=args.pitch_bias,
        cutoff_frequency=args.cutoff_frequency,
        max_target_step=args.max_target_step,
        action_gain=args.action_gain,
        initial_pose=args.initial_pose,
        start_frame=args.start_frame,
        action_clip=args.action_clip,
        joint_vel_scale=args.joint_vel_scale,
        safety_clip=not args.no_safety_clip,
        joint_limit_margin=args.joint_limit_margin,
        debug_interval_s=args.debug_interval_s,
        debug_top_k=args.debug_top_k,
        dry_run=args.dry_run,
        debug=args.debug,
    )
    runner.run()


if __name__ == "__main__":
    main()
