#!/usr/bin/env bash
set -e

python v2_rl_walk_mjlab.py \
  --onnx_model_path sway_t2_full.onnx \
  --motion_path A2_-_Sway_t2_stageii.npz \
  --duck_config_path ../duck_config.json \
  --start_frame 66 \
  --joint_vel_scale 0.10472 \
  --action_clip 3.0 \
  --action_gain 0.4 \
  --max_target_step 0.02 \
  --cutoff_frequency 5 \
  --debug \
  --debug_interval_s 0.5 \
  --log_path logs/test_run_vel0.log
