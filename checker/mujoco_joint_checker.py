from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

from joint_checker_common import (
    ACTION_ORDER_14,
    DEFAULT_REAL_XML,
    JOINT_ORDER_16,
    clamp_to_joint_limit,
    default_joint_limit,
    log_sample,
    open_csv_logger,
    print_joint_table,
    print_key_help,
)


def read_key(timeout_s: float) -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not readable:
        return None
    return sys.stdin.read(1)


def joint_qpos_addresses(mujoco, model) -> dict[str, int]:
    addresses: dict[str, int] = {}
    for joint_name in JOINT_ORDER_16:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        addresses[joint_name] = int(model.jnt_qposadr[joint_id])
    return addresses


def set_freejoint_pose(model, data, height: float) -> None:
    if model.nq < 7:
        return
    data.qpos[0:3] = np.array([0.0, 0.0, height])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])


def apply_targets(
    mujoco,
    model,
    data,
    qpos_addr: dict[str, int],
    targets: dict[str, float],
    root_height: float,
) -> None:
    set_freejoint_pose(model, data, root_height)
    for joint_name, address in qpos_addr.items():
        if joint_name.endswith("_backlash"):
            data.qpos[address] = 0.0
        else:
            data.qpos[address] = targets.get(joint_name, 0.0)
    mujoco.mj_forward(model, data)


def print_status(selected: str, target: float, current: float, index: int) -> None:
    print(
        f"\r[{index:02d}] {selected:16s} target={target:+.4f} "
        f"qpos={current:+.4f}",
        end="",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively set OpenDuck MuJoCo joint qpos without stepping physics. "
            "Use this to inspect zero position, offsets, and joint directions."
        )
    )
    parser.add_argument("--xml", default=str(DEFAULT_REAL_XML))
    parser.add_argument("--joint", default="left_hip_roll", choices=ACTION_ORDER_14)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Symmetric interactive limit. Defaults to the selected joint XML limit.",
    )
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--root_height", type=float, default=0.22)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--log", default="checker/logs/mujoco_joint_checker.csv")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print_joint_table()
        return 0

    import mujoco
    import mujoco.viewer

    xml_path = Path(args.xml).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    qpos_addr = joint_qpos_addresses(mujoco, model)

    missing = [joint_name for joint_name in ACTION_ORDER_14 if joint_name not in qpos_addr]
    if missing:
        raise RuntimeError(f"XML is missing action joints: {missing}")

    targets = {joint_name: 0.0 for joint_name in ACTION_ORDER_14}
    selected_index = ACTION_ORDER_14.index(args.joint)
    log_file, writer = open_csv_logger(args.log)

    period = 1.0 / max(args.rate, 1.0)
    print(f"XML: {xml_path}")
    print("MuJoCo checker sets qpos directly and does not call mj_step.")
    print_joint_table()
    print_key_help()

    old_terminal = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
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
                apply_targets(mujoco, model, data, qpos_addr, targets, args.root_height)
                current = float(data.qpos[qpos_addr[selected]])
                print_status(selected, targets[selected], current, selected_index)
                log_sample(
                    writer,
                    "mujoco",
                    selected,
                    selected,
                    targets[selected],
                    current,
                    raw=None,
                    event=event,
                )
                viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)
        if log_file is not None:
            log_file.close()
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
