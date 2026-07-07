cd ~/Open_Duck_Mini_Runtime/scripts
python v2_rl_walk_mjlab.py \
  --onnx_model_path sway_t2_full.onnx \
  --motion_path A2_-_Sway_t2_stageii.npz \
  --duck_config_path ../duck_config.json \
  --control_freq 50 \
  --max_target_step 0.04 \
  --action_gain 0.1 \
  --initial_pose motion_start \
  --debug
