import paho.mqtt.client as mqtt
import threading
import time
import numpy as np

from v2_rl_walk_mujoco import RLWalk
from mini_bdx_runtime.rl_utils import make_action_dict

# MQTT 基本配置
BROKER = "peien.xyz"        # 公共MQTT代理
PORT = 1883                      # 非加密端口
TOPIC = "2c:cf:67:56:d7:8c"      # 订阅主题（和小智端一致）

# 控制指令编号到命令向量的映射
X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]
CMD_DICT = {
    0: [0.0, 0.0, 0.0, 0, 0, 0, 0],                 # 停止
    1: [X_RANGE[1], 0.0, 0.0, 0, 0, 0, 0],          # 前进
    2: [X_RANGE[0], 0.0, 0.0, 0, 0, 0, 0],          # 后退
    3: [0.0, Y_RANGE[0], 0.0, 0, 0, 0, 0],          # 左移
    4: [0.0, Y_RANGE[1], 0.0, 0, 0, 0, 0],          # 右移
    5: [0.0, 0.0, YAW_RANGE[1]*0.5, 0, 0, 0, 0],    # 左转
    6: [0.0, 0.0, YAW_RANGE[0]*0.5, 0, 0, 0, 0],    # 右转
}
DEFAULT_CMD = [0.0, 0.0, 0.0, 0, 0, 0, 0]           # 停止

current_command = DEFAULT_CMD.copy()
cmd_lock = threading.Lock()
action_timer = None
ACTION_DURATION = 5    # 每条命令执行时长（秒）

def reset_to_stop():
    global current_command
    with cmd_lock:
        current_command = DEFAULT_CMD.copy()
    print("[动作] 5秒到，已自动归零（停止）")

def on_connect(client, userdata, flags, rc):
    print("已连接MQTT服务器")
    client.subscribe(TOPIC)
    print("已订阅主题", TOPIC)

def on_message(client, userdata, msg):
    global action_timer, current_command
    try:
        # 打印原始内容
        print(f"[MQTT] 收到原始消息：主题={msg.topic}，内容={msg.payload}")
        num = int(msg.payload.decode())
        print(f"[MQTT] 解析到指令编号：{num}")
        with cmd_lock:
            if num in CMD_DICT:
                current_command = CMD_DICT[num]
                print(f"[动作] 已切换为: {current_command}")
            else:
                current_command = DEFAULT_CMD.copy()
                print("[动作] 未知指令，已切换为停止")
        # 每次收到新命令都重启定时器
        if action_timer is not None:
            action_timer.cancel()
        if num != 0:  # 如果不是停止命令，计时5秒
            action_timer = threading.Timer(ACTION_DURATION, reset_to_stop)
            action_timer.start()
        else:
            # 如果是停止命令，立即归零并取消定时器
            if action_timer is not None:
                action_timer.cancel()
            reset_to_stop()
    except Exception as e:
        print("指令解析出错：", e)

def robot_control_loop():
    ONNX_MODEL_PATH = "/home/lxkj/Open_Duck_Mini_Runtime/scripts/BEST_WALK_ONNX_2.onnx"
    DUCK_CONFIG_PATH = "/home/lxkj/duck_config.json"
    rl_walk = RLWalk(
        onnx_model_path=ONNX_MODEL_PATH,
        duck_config_path=DUCK_CONFIG_PATH,
        action_scale=0.25,
        control_freq=50,
        commands=False
    )
    print("鸭子控制循环启动，等待小智指令…")
    try:
        while True:
            loop_start = time.time()
            with cmd_lock:
                rl_walk.last_commands = current_command.copy()
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
    except KeyboardInterrupt:
        print("中断退出，关闭舵机")
        rl_walk.hwi.turn_off()

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    # 启动机器人控制线程
    t = threading.Thread(target=robot_control_loop, daemon=True)
    t.start()
    # 主线程跑MQTT循环
    client.loop_forever()
