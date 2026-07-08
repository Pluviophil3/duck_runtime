import argparse
import json
import os
import sys
import time


HOME_DIR = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.dirname(SCRIPT_DIR)


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def print_config(config):
    print("duck_config.json:")
    print(json.dumps(config, indent=4, ensure_ascii=False))


def neutral_targets(joint_names):
    return {joint_name: 0.0 for joint_name in joint_names}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reload duck_config.json and command the robot to the neutral zero pose "
            "using the updated joints_offsets."
        )
    )
    parser.add_argument(
        "--duck_config_path",
        default=f"{HOME_DIR}/duck_config.json",
        help="Path to duck_config.json on the robot.",
    )
    parser.add_argument("--serial_port", default="/dev/ttyACM0")
    parser.add_argument(
        "--kp",
        type=float,
        default=2.0,
        help="Low proportional gain used while holding neutral.",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=0.0,
        help="Derivative gain used while holding neutral.",
    )
    parser.add_argument(
        "--hold_seconds",
        type=float,
        default=3.0,
        help="How long to keep sending the neutral command. Use 0 to send once.",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Only read and print duck_config.json; do not connect to motors.",
    )
    args = parser.parse_args()

    config = load_config(args.duck_config_path)
    print_config(config)

    if args.print_only:
        return

    sys.path.insert(0, os.path.join(RUNTIME_DIR, "mini_bdx_runtime"))
    from mini_bdx_runtime.duck_config import DuckConfig
    from mini_bdx_runtime.rustypot_position_hwi import HWI

    duck_config = DuckConfig(args.duck_config_path)
    hwi = HWI(duck_config, args.serial_port)
    target = neutral_targets(hwi.joints.keys())

    hwi.set_kps([args.kp] * len(hwi.joints))
    hwi.set_kds([args.kd] * len(hwi.joints))

    print("Commanding neutral zero pose with updated joints_offsets...")
    hwi.set_position_all(target)

    end_time = time.time() + max(0.0, args.hold_seconds)
    while time.time() < end_time:
        hwi.set_position_all(target)
        time.sleep(0.05)

    print("Done. Neutral command sent from duck_config.json.")


if __name__ == "__main__":
    main()
