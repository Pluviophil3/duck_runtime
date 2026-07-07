#!/usr/bin/env python3
# coding=utf-8
"""
command_test.py - 自动依次测试机械鸭子的各种典型控制命令（模拟手柄操作）

依赖 RLWalk、DuckConfig、OnnxInfer、make_action_dict
"""

import time
import numpy as np
from v2_rl_walk_mujoco import RLWalk
from mini_bdx_runtime.rl_utils import make_action_dict

# 路径根据实际情况修改
ONNX_MODEL_PATH = "/home/lxkj/Open_Duck_Mini_Runtime/scripts/BEST_WALK_ONNX_2.onnx"
DUCK_CONFIG_PATH = "/home/lxkj/duck_config.json"

# 命令区间（参考 xbox_controller.py）
X_RANGE = [-0.15, 0.15]     # 前进/后退
Y_RANGE = [-0.2, 0.2]       # 横向（通常不用）
YAW_RANGE = [-1.0, 1.0]     # 左右旋转

rl_walk = RLWalk(
    onnx_model_path=ONNX_MODEL_PATH,
    duck_config_path=DUCK_CONFIG_PATH,
    action_scale=0.25,
    control_freq=50,
    commands=False
)

def do_action(commands, duration=5):
    """
    给定命令，持续指定秒数
    """
    t0 = time.time()
    print(f"【TEST】动作: {commands} , 持续 {duration} 秒")
    while time.time() - t0 < duration:
        loop_start = time.time()
        rl_walk.last_commands = commands
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
        if rl_walk.action_filter is not None:
            rl_walk.action_filter.push(rl_walk.motor_targets)
            filtered_motor_targets = rl_walk.action_filter.get_filtered_action()
            if time.time() > 1.0:
                rl_walk.motor_targets = filtered_motor_targets
        rl_walk.prev_motor_targets = rl_walk.motor_targets.copy()
        head_targets = rl_walk.last_commands[3:] + rl_walk.motor_targets[5:9]
        rl_walk.motor_targets[5:9] = head_targets
        action_dict = make_action_dict(rl_walk.motor_targets, list(rl_walk.hwi.joints.keys()))
        rl_walk.hwi.set_position_all(action_dict)
        # 控制频率
        loop_duration = time.time() - loop_start
        time.sleep(max(0, 1 / rl_walk.control_freq - loop_duration))

try:
    print("【command_test.py】自动化命令测试开始")
    # 1. 前进5秒
    do_action([X_RANGE[1], 0.0, 0.0, 0, 0, 0, 0], duration=5)

    # 2. 后退5秒
    do_action([X_RANGE[0], 0.0, 0.0, 0, 0, 0, 0], duration=5)

    # 3. 原地左转5秒
    do_action([0.0, 0.0, YAW_RANGE[1]*0.5, 0, 0, 0, 0], duration=5)  # 0.5为适中速度

    # 4. 原地右转5秒
    do_action([0.0, 0.0, YAW_RANGE[0]*0.5, 0, 0, 0, 0], duration=5)

    # 5. 原地不动3秒
    do_action([0.0, 0.0, 0.0, 0, 0, 0, 0], duration=3)

    # 6. 横向平移左3秒
    do_action([0.0, Y_RANGE[0], 0.0, 0, 0, 0, 0], duration=3)

    # 7. 横向平移右3秒
    do_action([0.0, Y_RANGE[1], 0.0, 0, 0, 0, 0], duration=3)

    # 8. 测试头部 pitch/yaw/roll
    # 抬头3秒
    do_action([0, 0, 0, 0.5, 0, 0, 0], duration=3)
    # 摇头3秒
    do_action([0, 0, 0, 0, 0.5, 0, 0], duration=3)
    # 歪头3秒
    do_action([0, 0, 0, 0, 0, 0.5, 0], duration=3)

    # 9. 恢复站立
    do_action([0.0, 0.0, 0.0, 0, 0, 0, 0], duration=2)
    print("【command_test.py】自动测试完毕，机械鸭子已归位。")

except KeyboardInterrupt:
    print("【command_test.py】测试中断，正在关闭舵机…")
    rl_walk.hwi.turn_off()
    print("【command_test.py】退出。")
