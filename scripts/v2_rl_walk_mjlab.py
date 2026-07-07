import argparse
import os
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
        data = np.load(motion_path)
        self.joint_pos = data["joint_pos"].astype(np.float32)
        self.joint_vel = data["joint_vel"].astype(np.float32)
        self.body_quat_w = data["body_quat_w"].astype(np.float32)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])

        if self.joint_pos.shape[1] != len(JOINT_ORDER_16):
            raise ValueError(
                f"motion joint_pos dim {self.joint_pos.shape[1]} != {len(JOINT_ORDER_16)}"
            )
        if self.joint_vel.shape[1] != len(JOINT_ORDER_16):
            raise ValueError(
                f"motion joint_vel dim {self.joint_vel.shape[1]} != {len(JOINT_ORDER_16)}"
            )
        if self.body_quat_w.shape[1] <= ANCHOR_BODY_INDEX:
            raise ValueError("motion body_quat_w does not contain the anchor body")

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
        dry_run=False,
        debug=False,
    ):
        self.debug = debug
        self.dry_run = dry_run
        self.control_freq = control_freq
        self.max_target_step = max_target_step
        self.action_gain = action_gain

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
        self.motion_i = 0

        self.action_scale = np.array(
            [ACTION_SCALE_BY_JOINT[name] for name in ACTION_ORDER_14],
            dtype=np.float32,
        )
        self.motor_targets = np.zeros(ACTION_DIM, dtype=np.float32)
        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(control_freq, cutoff_frequency)

        self._check_runtime_order()
        if not dry_run:
            self.start()
            current_pos = self.hwi.get_present_positions()
            if current_pos is not None and len(current_pos) == ACTION_DIM:
                self.motor_targets = current_pos.astype(np.float32)

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

    def start(self):
        kps = [self.pid[0]] * ACTION_DIM
        kds = [self.pid[2]] * ACTION_DIM
        kps[5:9] = [8, 8, 8, 8]
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()
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
            vel_14 = vel_14.astype(np.float32)

        pos_by_name = dict(zip(ACTION_ORDER_14, pos_14))
        vel_by_name = dict(zip(ACTION_ORDER_14, vel_14))
        pos_16 = np.array([pos_by_name.get(name, 0.0) for name in JOINT_ORDER_16], dtype=np.float32)
        vel_16 = np.array([vel_by_name.get(name, 0.0) for name in JOINT_ORDER_16], dtype=np.float32)
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

    def _targets_from_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape {action.shape} != ({ACTION_DIM},)")
        target = action * self.action_scale * self.action_gain
        if self.max_target_step is not None and self.motion_i > 0:
            delta = np.clip(
                target - self.motor_targets,
                -self.max_target_step,
                self.max_target_step,
            )
            target = self.motor_targets + delta
        return target.astype(np.float32)

    def step(self):
        obs = self.get_obs()
        if obs is None:
            return False
        action = self.policy.infer(obs).astype(np.float32)
        target = self._targets_from_action(action)

        if self.action_filter is not None:
            self.action_filter.push(target)
            target = self.action_filter.get_filtered_action()

        self.last_action = action.copy()
        self.motor_targets = target.copy()

        if self.debug and self.motion_i % max(1, self.control_freq) == 0:
            print(
                "[mjlab] "
                f"frame={self.motion_i % self.motion.num_frames} "
                f"obs_norm={np.linalg.norm(obs):.3f} "
                f"action_abs_max={np.max(np.abs(action)):.3f} "
                f"action_gain={self.action_gain:.3f} "
                f"target_abs_max={np.max(np.abs(target)):.3f}"
            )

        if self.hwi is not None:
            self.hwi.set_position_all(make_action_dict(target, ACTION_ORDER_14))

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
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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
        dry_run=args.dry_run,
        debug=args.debug,
    )
    runner.run()


if __name__ == "__main__":
    main()
