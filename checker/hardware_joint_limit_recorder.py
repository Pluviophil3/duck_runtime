from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

from joint_checker_common import (
    ACTION_ORDER_14,
    DEFAULT_DUCK_CONFIG,
    clamp_to_joint_limit,
    default_joint_limit,
    print_joint_table,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "logs" / "hardware_joint_limits.json"


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
        print(f"\nCould not read {joint_name} raw position: {exc}")
        return None


def empty_results() -> dict[str, Any]:
    return {
        "format": "open_duck_hardware_joint_limits_v1",
        "updated_at": "",
        "limits": {
            joint_name: {
                "min_logical_rad": None,
                "max_logical_rad": None,
                "samples": [],
            }
            for joint_name in ACTION_ORDER_14
        },
    }


def load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_results()
    with path.open("r") as file:
        data = json.load(file)
    base = empty_results()
    for joint_name, entry in data.get("limits", {}).items():
        if joint_name not in base["limits"]:
            continue
        base_entry = base["limits"][joint_name]
        base_entry["min_logical_rad"] = entry.get("min_logical_rad")
        base_entry["max_logical_rad"] = entry.get("max_logical_rad")
        base_entry["samples"] = dedupe_samples(entry.get("samples", []))
    return base


def dedupe_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, float]] = set()
    unique: list[dict[str, Any]] = []
    for sample in samples:
        joint_name = str(sample.get("joint", ""))
        logical = sample.get("logical_rad")
        if logical is None:
            continue
        key = (joint_name, round(float(logical), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def record_limit_sample(
    results: dict[str, Any],
    joint_name: str,
    target: float,
    logical: float,
    raw: float | None,
) -> None:
    entry = results["limits"][joint_name]
    current_min = entry["min_logical_rad"]
    current_max = entry["max_logical_rad"]
    if current_min is None or logical < float(current_min):
        entry["min_logical_rad"] = logical
    if current_max is None or logical > float(current_max):
        entry["max_logical_rad"] = logical

    sample = {
        "time_s": round(time.time(), 6),
        "joint": joint_name,
        "target_rad": round(target, 8),
        "logical_rad": round(logical, 8),
        "raw_rad": None if raw is None else round(raw, 8),
    }
    entry["samples"].append(sample)
    entry["samples"] = dedupe_samples(entry["samples"])


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with path.open("w") as file:
        json.dump(results, file, indent=2, sort_keys=True)
        file.write("\n")


def print_key_help() -> None:
    print("Keys:")
    print("  w / +       increase selected joint target")
    print("  s / -       decrease selected joint target")
    print("  r           record current logical as min/max sample")
    print("  0           reset selected joint target to zero")
    print("  space       reset all joint targets to zero")
    print("  . / n       next joint")
    print("  , / p       previous joint")
    print("  number      select joint index 0-9; use . for 10+")
    print("  q           save and quit")


def print_status(
    joint_name: str,
    target: float,
    logical_pos: float | None,
    raw_pos: float | None,
    results: dict[str, Any],
) -> None:
    logical_text = "None" if logical_pos is None else f"{logical_pos:+.4f}"
    raw_text = "None" if raw_pos is None else f"{raw_pos:+.4f}"
    entry = results["limits"][joint_name]
    min_value = entry["min_logical_rad"]
    max_value = entry["max_logical_rad"]
    min_text = "None" if min_value is None else f"{float(min_value):+.4f}"
    max_text = "None" if max_value is None else f"{float(max_value):+.4f}"
    print(
        f"\r{joint_name} target={target:+.4f} rad "
        f"logical={logical_text} rad raw={raw_text} rad "
        f"recorded=[{min_text}, {max_text}]",
        end="",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively command real OpenDuck joints and record observed logical "
            "joint limits to a de-duplicated JSON file."
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
        help="Symmetric interactive command limit. Defaults to the current XML limit.",
    )
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--neutral_hold_seconds", type=float, default=1.0)
    parser.add_argument("--turn_off_on_exit", action="store_true")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON file to update with recorded logical min/max limits.",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print_joint_table()
        return 0

    output_path = Path(args.output).expanduser().resolve()
    results = load_results(output_path)

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

    hwi.set_kps([args.kp] * len(hwi.joints))
    hwi.set_kds([args.kd] * len(hwi.joints))
    hwi.io.enable_torque(list(hwi.joints.values()))

    print(f"Reading config from: {duck_config_path}")
    print(f"Writing limits to: {output_path}")
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
        print("Interactive hardware limit recording started.")
        while True:
            selected = ACTION_ORDER_14[selected_index]
            limit = args.limit if args.limit is not None else default_joint_limit(
                selected, args.margin
            )
            key = read_key(period)
            record_requested = False
            if key in ("q", "Q"):
                break
            if key in ("+", "=", "w", "W"):
                targets[selected] = min(limit, targets[selected] + args.step)
                targets[selected] = clamp_to_joint_limit(
                    selected, targets[selected], args.margin
                )
            elif key in ("-", "_", "s", "S"):
                targets[selected] = max(-limit, targets[selected] - args.step)
                targets[selected] = clamp_to_joint_limit(
                    selected, targets[selected], args.margin
                )
            elif key in ("r", "R"):
                record_requested = True
            elif key == "0":
                targets[selected] = 0.0
            elif key == " ":
                for joint_name in targets:
                    targets[joint_name] = 0.0
            elif key in (".", "n", "N"):
                selected_index = (selected_index + 1) % len(ACTION_ORDER_14)
            elif key in (",", "p", "P"):
                selected_index = (selected_index - 1) % len(ACTION_ORDER_14)
            elif key is not None and key.isdigit():
                selected_index = min(int(key), len(ACTION_ORDER_14) - 1)

            selected = ACTION_ORDER_14[selected_index]
            hwi.set_position(selected, targets[selected])
            logical_pos = logical_position_for_joint(hwi, selected)
            raw_pos = raw_position_for_joint(hwi, selected)
            if record_requested and logical_pos is not None:
                record_limit_sample(
                    results,
                    selected,
                    targets[selected],
                    logical_pos,
                    raw_pos,
                )
                write_results(output_path, results)
                print(f"\nRecorded {selected}: logical={logical_pos:+.6f} rad")
            print_status(selected, targets[selected], logical_pos, raw_pos, results)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)
        write_results(output_path, results)
        if args.turn_off_on_exit:
            hwi.io.disable_torque(list(hwi.joints.values()))
            print("\nDisabled torque for all checker joints.")
        else:
            print("\nExit. Motors keep their last commanded targets.")
        print(f"Saved de-duplicated limits to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
