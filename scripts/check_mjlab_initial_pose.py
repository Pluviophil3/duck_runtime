import argparse
import ast
import os
import sys
import time

import numpy as np


HOME_DIR = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.dirname(SCRIPT_DIR)


def load_runtime_orders():
    script_path = os.path.join(SCRIPT_DIR, "v2_rl_walk_mjlab.py")
    tree = ast.parse(open(script_path, "r").read())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (
                "JOINT_ORDER_16",
                "ACTION_ORDER_14",
            ):
                constants[target.id] = ast.literal_eval(node.value)
    return constants["JOINT_ORDER_16"], constants["ACTION_ORDER_14"]


JOINT_ORDER_16, ACTION_ORDER_14 = load_runtime_orders()


def load_motion_start_targets(motion_path):
    motion = np.load(motion_path)
    joint_pos_16 = motion["joint_pos"][0].astype(np.float32)
    if joint_pos_16.shape != (len(JOINT_ORDER_16),):
        raise ValueError(
            f"motion joint_pos[0] shape {joint_pos_16.shape} does not match "
            f"{len(JOINT_ORDER_16)} joints"
        )
    values_by_name = dict(zip(JOINT_ORDER_16, joint_pos_16))
    return {name: float(values_by_name[name]) for name in ACTION_ORDER_14}


def zero_targets():
    return {name: 0.0 for name in ACTION_ORDER_14}


def target_for_mode(mode, motion_path):
    if mode == "zero":
        return zero_targets()
    if mode == "motion_start":
        return load_motion_start_targets(motion_path)
    raise ValueError(f"Unsupported pose mode {mode!r}")


def print_table(title, values):
    print(title)
    for name in ACTION_ORDER_14:
        print(f"  {name:16s} {values[name]: .6f}")


def read_positions_by_name(hwi):
    values = hwi.get_present_positions()
    if values is None:
        return None
    return {name: float(value) for name, value in zip(ACTION_ORDER_14, values)}


def print_error(current, target):
    errors = {name: current[name] - target[name] for name in ACTION_ORDER_14}
    print_table("current position (rad):", current)
    print_table("error current-target (rad):", errors)
    max_name = max(errors, key=lambda name: abs(errors[name]))
    print(f"max_abs_error: {max_name} {abs(errors[max_name]):.6f} rad")


def main():
    parser = argparse.ArgumentParser(
        description="Check or hold a mjlab-aligned initial pose without loading a policy."
    )
    parser.add_argument(
        "--pose",
        choices=("motion_start", "zero"),
        default="motion_start",
        help="motion_start matches tracking play reset; zero matches mjlab default joint position.",
    )
    parser.add_argument(
        "--motion_path",
        default="A2_-_Sway_t2_stageii.npz",
        help="Required for --pose motion_start.",
    )
    parser.add_argument(
        "--duck_config_path",
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("--serial_port", default="/dev/ttyACM0")
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--hold_seconds", type=float, default=0.0)
    parser.add_argument(
        "--command",
        action="store_true",
        help="Actually command the target pose. Without this, only prints target/current.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print target pose without connecting to hardware.",
    )
    args = parser.parse_args()

    motion_path = args.motion_path
    if not os.path.isabs(motion_path):
        motion_path = os.path.join(os.path.dirname(__file__), motion_path)
    target = target_for_mode(args.pose, motion_path)

    print_table(f"target pose '{args.pose}' (rad):", target)
    if args.dry_run:
        return

    sys.path.insert(0, os.path.join(RUNTIME_DIR, "mini_bdx_runtime"))
    from mini_bdx_runtime.duck_config import DuckConfig
    from mini_bdx_runtime.rustypot_position_hwi import HWI

    duck_config = DuckConfig(args.duck_config_path)
    hwi = HWI(duck_config, args.serial_port)
    if tuple(hwi.joints.keys()) != ACTION_ORDER_14:
        raise ValueError(
            "HWI joint order does not match mjlab action order:\n"
            f"HWI:   {tuple(hwi.joints.keys())}\n"
            f"mjlab: {ACTION_ORDER_14}"
        )

    current = read_positions_by_name(hwi)
    if current is not None:
        print_error(current, target)

    if not args.command:
        print("No motor command sent. Add --command to move/hold this pose.")
        return

    kps = [args.kp] * len(ACTION_ORDER_14)
    kds = [args.kd] * len(ACTION_ORDER_14)
    hwi.set_kps(kps)
    hwi.set_kds(kds)
    hwi.set_position_all(target)
    print(f"Commanded target pose with kp={args.kp}, kd={args.kd}")

    end_time = time.time() + max(0.0, args.hold_seconds)
    while time.time() < end_time:
        hwi.set_position_all(target)
        time.sleep(0.05)

    current = read_positions_by_name(hwi)
    if current is not None:
        print_error(current, target)


if __name__ == "__main__":
    main()
