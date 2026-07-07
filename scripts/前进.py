#!/usr/bin/env python3
# coding=utf-8
"""
前进.py - 模拟手柄持续推杆前进，控制机械鸭子持续前进
依赖项目已有 RLWalk、DuckConfig、OnnxInfer 等模块
"""

import time
import numpy as np

from v2_rl_walk_mujoco import RLWalk    # 路径根据你实际项目结构来
from mini_bdx_runtime.rl_utils import make_action_dict

# 路径根据实际情况修改
ONNX_MODEL_PATH = "/home/lxkj/Open_Duck_Mini_Runtime/scripts/BEST_WALK_ONNX_2.onnx"
DUCK_CONFIG_PATH = "/home/lxkj/duck_config.json"

# 1. 初始化 RLWalk 实例（关闭手柄命令输入，只靠脚本提供命令）
rl_walk = RLWalk(
    onnx_model_path=ONNX_MODEL_PATH,
    duck_config_path=DUCK_CONFIG_PATH,
    action_scale=0.25,
    control_freq=50,
    commands=False      # 不启用手柄
)

print("【前进.py】启动：机械鸭子将持续前进！")

# 2. 持续提供“前进最大值”命令，等价于摇杆推满
# last_commands: [前进速度, 横移速度, 角速度, 头pitch, 头yaw, 头roll, ...]
commands = [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 0.15可根据X_RANGE最大值设定

try:
    while True:
        loop_start = time.time()
        rl_walk.last_commands = commands

        # === RL主控流程核心环节 ===
        obs = rl_walk.get_obs()
        if obs is None:
            continue
        rl_walk.imitation_i += 1 * (rl_walk.phase_frequency_factor + rl_walk.phase_frequency_factor_offset)
        rl_walk.imitation_i %= rl_walk.PRM.nb_steps_in_period
        rl_walk.imitation_phase = np.array([
            np.cos(rl_walk.imitation_i / rl_walk.PRM.nb_steps_in_period * 2 * np.pi),
            np.sin(rl_walk.imitation_i / rl_walk.PRM.nb_steps_in_period * 2 * np.pi)
        ])
        action = rl_walk.policy.infer(obs)
        rl_walk.last_last_last_action = rl_walk.last_last_action.copy()
        rl_walk.last_last_action = rl_walk.last_action.copy()
        rl_walk.last_action = action.copy()
        rl_walk.motor_targets = rl_walk.init_pos + action * rl_walk.action_scale
        # 低通滤波处理（如有配置）
        if rl_walk.action_filter is not None:
            rl_walk.action_filter.push(rl_walk.motor_targets)
            filtered_motor_targets = rl_walk.action_filter.get_filtered_action()
            if time.time() > 1.0:
                rl_walk.motor_targets = filtered_motor_targets
        rl_walk.prev_motor_targets = rl_walk.motor_targets.copy()
        # 头部动作（此处为零）
        head_targets = rl_walk.last_commands[3:] + rl_walk.motor_targets[5:9]
        rl_walk.motor_targets[5:9] = head_targets
        action_dict = make_action_dict(rl_walk.motor_targets, list(rl_walk.hwi.joints.keys()))
        rl_walk.hwi.set_position_all(action_dict)
        # 控制频率
        loop_duration = time.time() - loop_start
        time.sleep(max(0, 1 / rl_walk.control_freq - loop_duration))

except KeyboardInterrupt:
    print("【前进.py】已中断，正在关闭舵机…")
    rl_walk.hwi.turn_off()
    print("【前进.py】退出。")
