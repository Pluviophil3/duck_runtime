from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import TextIO


def find_workspace_root() -> Path:
    checker_parent = Path(__file__).resolve().parents[1]
    if (checker_parent / "unitree_rl_mjlab").exists():
        return checker_parent
    if (checker_parent / "duck_runtime").exists():
        return checker_parent
    return checker_parent


def default_duck_config_path(root: Path) -> Path:
    if (root / "duck_runtime" / "duck_config.json").exists():
        return root / "duck_runtime" / "duck_config.json"
    return root / "duck_config.json"


REPO_ROOT = find_workspace_root()
DEFAULT_REAL_XML = (
    REPO_ROOT
    / "unitree_rl_mjlab"
    / "src"
    / "assets"
    / "robots"
    / "open_duck_mini_v2"
    / "xmls"
    / "open_duck_mini_v2_real.xml"
)
DEFAULT_DUCK_CONFIG = default_duck_config_path(REPO_ROOT)

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

ACTION_ORDER_14 = tuple(
    joint_name for joint_name in JOINT_ORDER_16 if "antenna" not in joint_name
)

JOINT_LIMITS = {
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

LOG_FIELDS = (
    "time_s",
    "source",
    "selected_joint",
    "joint",
    "target_rad",
    "current_rad",
    "raw_rad",
    "event",
)


def clamp_to_joint_limit(joint_name: str, value: float, margin: float = 0.0) -> float:
    lower, upper = JOINT_LIMITS[joint_name]
    return min(max(value, lower + margin), upper - margin)


def default_joint_limit(joint_name: str, margin: float = 0.0) -> float:
    lower, upper = JOINT_LIMITS[joint_name]
    return max(abs(lower + margin), abs(upper - margin))


def open_csv_logger(path: str | Path | None) -> tuple[TextIO | None, csv.DictWriter | None]:
    if path is None:
        return None, None
    log_path = Path(path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="", buffering=1)
    writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    writer.writeheader()
    return log_file, writer


def log_sample(
    writer: csv.DictWriter | None,
    source: str,
    selected_joint: str,
    joint: str,
    target: float,
    current: float | None,
    raw: float | None = None,
    event: str = "",
) -> None:
    if writer is None:
        return
    writer.writerow(
        {
            "time_s": f"{time.time():.6f}",
            "source": source,
            "selected_joint": selected_joint,
            "joint": joint,
            "target_rad": f"{target:.8f}",
            "current_rad": "" if current is None else f"{current:.8f}",
            "raw_rad": "" if raw is None else f"{raw:.8f}",
            "event": event,
        }
    )


def print_joint_table() -> None:
    for index, joint_name in enumerate(ACTION_ORDER_14):
        lower, upper = JOINT_LIMITS[joint_name]
        print(f"{index:2d}: {joint_name:16s} limit=[{lower:+.3f}, {upper:+.3f}]")


def print_key_help() -> None:
    print("Keys:")
    print("  w / +       increase selected joint target")
    print("  s / -       decrease selected joint target")
    print("  0           reset selected joint target to zero")
    print("  space       reset all joint targets to zero")
    print("  . / n       next joint")
    print("  , / p       previous joint")
    print("  number      select joint index 0-9; use . for 10+")
    print("  q           quit")
