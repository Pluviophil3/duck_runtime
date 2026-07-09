#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ONNX_MODEL_PATH="${ONNX_MODEL_PATH:-open_duck_real_backlash.onnx}"
MOTION_PATH="${MOTION_PATH:-A2_-_Sway_t2_stageii.npz}"
DUCK_CONFIG_PATH="${DUCK_CONFIG_PATH:-../duck_config.json}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyACM0}"
CONTROL_FREQ="${CONTROL_FREQ:-50}"
MAX_TARGET_STEP="${MAX_TARGET_STEP:-0.08}"
INITIAL_POSE="${INITIAL_POSE:-motion_start}"
ACTION_GAIN="${ACTION_GAIN:-1.0}"
JOINT_VEL_SCALE="${JOINT_VEL_SCALE:-1.0}"
P_GAIN="${P_GAIN:-30}"
I_GAIN="${I_GAIN:-0}"
D_GAIN="${D_GAIN:-0}"

exec python v2_rl_walk_mjlab.py \
  --onnx_model_path "${ONNX_MODEL_PATH}" \
  --motion_path "${MOTION_PATH}" \
  --duck_config_path "${DUCK_CONFIG_PATH}" \
  --serial_port "${SERIAL_PORT}" \
  --control_freq "${CONTROL_FREQ}" \
  --max_target_step "${MAX_TARGET_STEP}" \
  --initial_pose "${INITIAL_POSE}" \
  --action_gain "${ACTION_GAIN}" \
  --joint_vel_scale "${JOINT_VEL_SCALE}" \
  -p "${P_GAIN}" \
  -i "${I_GAIN}" \
  -d "${D_GAIN}" \
  --debug \
  "$@"
