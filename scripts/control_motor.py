import argparse
import os
import select
import sys
import termios
import time
import tty


HOME_DIR = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.dirname(SCRIPT_DIR)


def read_key(timeout):
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return None
    return sys.stdin.read(1)


def logical_position_for_joint(hwi, joint_name):
    positions = hwi.get_present_positions()
    if positions is None:
        return None
    joint_names = list(hwi.joints.keys())
    return float(positions[joint_names.index(joint_name)])


def raw_position_for_joint(hwi, joint_name):
    joint_id = hwi.joints[joint_name]
    try:
        return float(hwi.io.read_present_position([joint_id])[0])
    except Exception as exc:
        print(f"Could not read {joint_name} raw position: {exc}")
        return None


def print_status(joint_name, target, logical_pos, raw_pos):
    logical_text = "None" if logical_pos is None else f"{logical_pos:+.4f}"
    raw_text = "None" if raw_pos is None else f"{raw_pos:+.4f}"
    print(
        f"\r{joint_name} target={target:+.4f} rad "
        f"current={logical_text} rad raw={raw_text} rad",
        end="",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interactively control one motor from the neutral pose."
    )
    parser.add_argument("joint_name", help="Joint name, for example neck_pitch.")
    parser.add_argument(
        "--duck_config_path",
        default=f"{HOME_DIR}/duck_config.json",
        help="Path to duck_config.json on the robot.",
    )
    parser.add_argument("--serial_port", default="/dev/ttyACM0")
    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Target angle step in radians for each key press.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=0.8,
        help="Absolute target limit in radians around neutral.",
    )
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Status print and command refresh rate in Hz.",
    )
    parser.add_argument(
        "--neutral_hold_seconds",
        type=float,
        default=1.0,
        help="How long to hold the neutral pose before interactive control.",
    )
    parser.add_argument(
        "--turn_off_on_exit",
        action="store_true",
        help="Disable torque on the selected motor when exiting.",
    )
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(RUNTIME_DIR, "mini_bdx_runtime"))
    from mini_bdx_runtime.duck_config import DuckConfig
    from mini_bdx_runtime.rustypot_position_hwi import HWI

    duck_config = DuckConfig(args.duck_config_path)
    hwi = HWI(duck_config, args.serial_port)

    if args.joint_name not in hwi.joints:
        print(f"Unknown joint: {args.joint_name}")
        print("Available joints:")
        for joint_name in hwi.joints:
            print(f"  {joint_name}")
        return 1

    joint_id = hwi.joints[args.joint_name]
    neutral = {joint_name: 0.0 for joint_name in hwi.joints}
    target = 0.0
    period = 1.0 / max(args.rate, 1.0)

    hwi.set_kps([args.kp] * len(hwi.joints))
    hwi.set_kds([args.kd] * len(hwi.joints))
    hwi.io.enable_torque(list(hwi.joints.values()))

    print(f"Reading config from: {args.duck_config_path}")
    print(f"Controlling {args.joint_name} motor id={joint_id}")
    print(f"step={args.step:.4f} rad, limit=+/-{args.limit:.4f} rad")
    print("Keys: + or w increase, - or s decrease, 0 neutral, q quit")
    print("Moving to neutral pose...")

    end_time = time.time() + max(0.0, args.neutral_hold_seconds)
    while time.time() < end_time:
        hwi.set_position_all(neutral)
        time.sleep(0.05)

    old_terminal = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("Interactive control started.")
        while True:
            key = read_key(period)
            if key in ("q", "Q"):
                break
            if key in ("+", "=", "w", "W"):
                target = min(args.limit, target + args.step)
            elif key in ("-", "_", "s", "S"):
                target = max(-args.limit, target - args.step)
            elif key == "0":
                target = 0.0

            hwi.set_position(args.joint_name, target)
            logical_pos = logical_position_for_joint(hwi, args.joint_name)
            raw_pos = raw_position_for_joint(hwi, args.joint_name)
            print_status(args.joint_name, target, logical_pos, raw_pos)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)
        print("")
        if args.turn_off_on_exit:
            hwi.io.disable_torque([joint_id])
            print(f"Disabled torque for {args.joint_name}.")
        else:
            print(f"Exit. {args.joint_name} is still holding target {target:+.4f} rad.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
