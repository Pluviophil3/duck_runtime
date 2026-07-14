from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

from joint_checker_common import (
    ACTION_ORDER_14,
    DEFAULT_DUCK_CONFIG,
    clamp_to_joint_limit,
    default_joint_limit,
    log_sample,
    open_csv_logger,
    print_joint_table,
    print_key_help,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def find_runtime_package_dir() -> Path:
    candidates = (
        REPO_ROOT / "duck_runtime" / "mini_bdx_runtime",
        REPO_ROOT / "mini_bdx_runtime",
    )
    for candidate in candidates:
        if (candidate / "mini_bdx_runtime").exists():
            return candidate
    raise RuntimeError(
        "Could not find mini_bdx_runtime package. Run from duck_amp, or copy "
        "checker/ into the duck_runtime root next to mini_bdx_runtime/."
    )


def read_key(timeout_s: float) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    return sys.stdin.read(1)


def import_runtime_hwi():
    sys.path.insert(0, str(find_runtime_package_dir()))
    from mini_bdx_runtime.duck_config import DuckConfig
    from mini_bdx_runtime.rustypot_position_hwi import HWI

    return DuckConfig, HWI


def logical_position_for_joint(hwi, joint_name: str) -> float | None:
    positions = hwi.get_present_positions()
    if positions is None:
        return None
    joint_names = list(hwi.joints.keys())
    return float(positions[joint_names.index(joint_name)])


def raw_position_for_joint(hwi, joint_name: str) -> float | None:
    joint_id = hwi.joints[joint_name]
    try:
        return float(hwi.io.read_present_position([joint_id])[0])
    except Exception as exc:
        print(f"Could not read {joint_name} raw position: {exc}")
        return None


def print_status(joint_name: str, target: float, logical_pos: float | None, raw_pos: float | None) -> None:
    logical_text = "None" if logical_pos is None else f"{logical_pos:+.4f}"
    raw_text = "None" if raw_pos is None else f"{raw_pos:+.4f}"
    print(
        f"\r{joint_name} target={target:+.4f} rad "
        f"logical={logical_text} rad raw={raw_text} rad",
        end="",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively command one real OpenDuck joint using the same joint order "
            "as the MuJoCo checker. Compare the logged target/current trajectory "
            "against checker/mujoco_joint_checker.py."
        )
    )
    parser.add_argument("--joint", default="left_hip_roll", choices=ACTION_ORDER_14)
    parser.add_argument("--duck_config_path", default=str(DEFAULT_DUCK_CONFIG))
    parser.add_argument("--serial_port", default="/dev/ttyACM0")
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Symmetric interactive limit. Defaults to the selected joint XML limit.",
    )
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--neutral_hold_seconds", type=float, default=1.0)
    parser.add_argument("--turn_off_on_exit", action="store_true")
    parser.add_argument("--log", default="checker/logs/hardware_joint_checker.csv")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print_joint_table()
        return 0

    DuckConfig, HWI = import_runtime_hwi()
    duck_config_path = os.path.expanduser(args.duck_config_path)
    duck_config = DuckConfig(duck_config_path)
    hwi = HWI(duck_config, args.serial_port)

    runtime_joint_order = tuple(hwi.joints.keys())
    if runtime_joint_order != ACTION_ORDER_14:
        raise RuntimeError(
            "HWI joint order does not match checker action order:\n"
            f"HWI:     {runtime_joint_order}\n"
            f"checker: {ACTION_ORDER_14}"
        )

    targets = {joint_name: 0.0 for joint_name in ACTION_ORDER_14}
    selected_index = ACTION_ORDER_14.index(args.joint)
    period = 1.0 / max(args.rate, 1.0)
    log_file, writer = open_csv_logger(args.log)

    hwi.set_kps([args.kp] * len(hwi.joints))
    hwi.set_kds([args.kd] * len(hwi.joints))
    hwi.io.enable_torque(list(hwi.joints.values()))

    print(f"Reading config from: {duck_config_path}")
    print(f"serial_port={args.serial_port}, kp={args.kp}, kd={args.kd}")
    print_joint_table()
    print_key_help()
    print("Moving all commanded joints to logical zero...")
    end_time = time.time() + max(0.0, args.neutral_hold_seconds)
    while time.time() < end_time:
        hwi.set_position_all(targets)
        time.sleep(0.05)

    old_terminal = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("Interactive hardware control started.")
        while True:
            selected = ACTION_ORDER_14[selected_index]
            limit = args.limit if args.limit is not None else default_joint_limit(
                selected, args.margin
            )
            key = read_key(period)
            event = ""
            if key in ("q", "Q"):
                break
            if key in ("+", "=", "w", "W"):
                targets[selected] = min(limit, targets[selected] + args.step)
                targets[selected] = clamp_to_joint_limit(
                    selected, targets[selected], args.margin
                )
                event = "increase"
            elif key in ("-", "_", "s", "S"):
                targets[selected] = max(-limit, targets[selected] - args.step)
                targets[selected] = clamp_to_joint_limit(
                    selected, targets[selected], args.margin
                )
                event = "decrease"
            elif key == "0":
                targets[selected] = 0.0
                event = "zero_selected"
            elif key == " ":
                for joint_name in targets:
                    targets[joint_name] = 0.0
                event = "zero_all"
            elif key in (".", "n", "N"):
                selected_index = (selected_index + 1) % len(ACTION_ORDER_14)
                event = "next_joint"
            elif key in (",", "p", "P"):
                selected_index = (selected_index - 1) % len(ACTION_ORDER_14)
                event = "previous_joint"
            elif key is not None and key.isdigit():
                selected_index = min(int(key), len(ACTION_ORDER_14) - 1)
                event = "select_joint"

            selected = ACTION_ORDER_14[selected_index]
            hwi.set_position(selected, targets[selected])
            logical_pos = logical_position_for_joint(hwi, selected)
            raw_pos = raw_position_for_joint(hwi, selected)
            print_status(selected, targets[selected], logical_pos, raw_pos)
            log_sample(
                writer,
                "hardware",
                selected,
                selected,
                targets[selected],
                logical_pos,
                raw_pos,
                event=event,
            )
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)
        if args.turn_off_on_exit:
            hwi.io.disable_torque(list(hwi.joints.values()))
            print("\nDisabled torque for all checker joints.")
        else:
            print("\nExit. Motors keep their last commanded targets.")
        if log_file is not None:
            log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
