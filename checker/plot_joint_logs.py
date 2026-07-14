from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.expanduser().open(newline="") as f:
        return list(csv.DictReader(f))


def series(rows: list[dict[str, str]], joint: str, field: str) -> tuple[list[float], list[float]]:
    filtered = [row for row in rows if row["joint"] == joint and row[field] != ""]
    if not filtered:
        return [], []
    t0 = float(filtered[0]["time_s"])
    return (
        [float(row["time_s"]) - t0 for row in filtered],
        [float(row[field]) for row in filtered],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot checker CSV target/current trajectories.")
    parser.add_argument("logs", nargs="+", help="CSV logs from mujoco or hardware checker.")
    parser.add_argument("--joint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    for log_path_text in args.logs:
        log_path = Path(log_path_text)
        rows = load_rows(log_path)
        label = log_path.stem
        for field, style in (("target_rad", "--"), ("current_rad", "-")):
            xs, ys = series(rows, args.joint, field)
            if xs:
                plt.plot(xs, ys, style, label=f"{label}:{field}")

    plt.xlabel("time since first sample (s)")
    plt.ylabel("angle (rad)")
    plt.title(args.joint)
    plt.grid(True)
    plt.legend()
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=160)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
