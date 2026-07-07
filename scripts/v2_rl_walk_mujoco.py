import time
import pickle
import threading
import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
from mini_bdx_runtime.duck_config import DuckConfig
import paho.mqtt.client as mqtt
import os

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=0.25,
        commands=False,
        pitch_bias=0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
        # MQTT配置参数
        mqtt_broker="peien.xyz",
        mqtt_port=1883,
        mqtt_topic="2c:cf:67:56:d7:8c",
        action_duration=5,
        debug=False,  # 新增调试模式参数
    ):

        self.duck_config = DuckConfig(config_json_path=duck_config_path)
        self.debug = debug  # 启用调试输出
        self.commands = commands
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14
        self.max_motor_velocity = 5.24  # rad/s

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        self.hwi = HWI(self.duck_config, serial_port)
        if self.debug:
            print(f"[调试] 连接的电机: {list(self.hwi.joints.keys())}")

        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = FeetContacts()

        # Scales
        self.action_scale = action_scale

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = list(self.hwi.init_pos.values())
        if self.debug:
            print(f"[调试] 初始电机位置: {self.init_pos}")

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused
        if self.debug:
            print(f"[调试] 初始暂停状态: {self.paused}")

        self.command_freq = 20  # hz
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)
            if self.debug:
                print("[调试] 手柄控制已启用")

        # Reference motion
        self.PRM = PolyReferenceMotion("./polynomial_coefficients.pkl")
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        # Optional expression features
        if self.duck_config.eyes:
            self.eyes = Eyes()
        if self.duck_config.projector:
            self.projector = Projector()
        if self.duck_config.speaker:
            self.sounds = Sounds(
                volume=1.0, sound_directory="../mini_bdx_runtime/assets/"
            )
        if self.duck_config.antennas:
            self.antennas = Antennas()

        # MQTT相关初始化
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.action_duration = action_duration
        self.mqtt_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.cmd_lock = threading.Lock()
        self.mqtt_timer = None
        self.setup_mqtt()

    def setup_mqtt(self):
        """初始化MQTT客户端并启动线程"""
        def on_connect(client, userdata, flags, rc):
            print(f"MQTT连接成功 (代码: {rc})")
            client.subscribe(self.mqtt_topic)
            if self.debug:
                print(f"[调试] 已订阅MQTT主题: {self.mqtt_topic}")
        
        def on_message(client, userdata, msg):
            """处理语音指令"""
            try:
                cmd_num = int(msg.payload.decode())
                # 语音指令映射
                X_RANGE = [-0.15, 0.15]
                Y_RANGE = [-0.2, 0.2]
                YAW_RANGE = [-1.0, 1.0]
                cmd_dict = {
                    0: [0.0, 0.0, 0.0, 0, 0, 0, 0],                 # 停止
                    1: [X_RANGE[1], 0.0, 0.0, 0, 0, 0, 0],          # 前进
                    2: [X_RANGE[0], 0.0, 0.0, 0, 0, 0, 0],          # 后退
                    3: [0.0, Y_RANGE[0], 0.0, 0, 0, 0, 0],          # 左移
                    4: [0.0, Y_RANGE[1], 0.0, 0, 0, 0, 0],          # 右移
                    5: [0.0, 0.0, YAW_RANGE[1]*0.5, 0, 0, 0, 0],    # 左转
                    6: [0.0, 0.0, YAW_RANGE[0]*0.5, 0, 0, 0, 0],    # 右转
                }
                with self.cmd_lock:
                    self.mqtt_command = cmd_dict.get(cmd_num, cmd_dict[0])
                
                print(f"[MQTT] 收到指令: {cmd_num}, 解析为: {self.mqtt_command}")
                
                # 重置超时定时器
                if self.mqtt_timer:
                    self.mqtt_timer.cancel()
                if cmd_num != 0:
                    self.mqtt_timer = threading.Timer(
                        self.action_duration, 
                        self._reset_mqtt_command
                    )
                    self.mqtt_timer.start()
            except Exception as e:
                print(f"MQTT指令解析错误: {e}")
        
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        
        # 启动MQTT线程
        mqtt_thread = threading.Thread(
            target=self.mqtt_client.loop_forever, 
            daemon=True
        )
        mqtt_thread.start()
        if self.debug:
            print("[调试] MQTT线程已启动")

    def _reset_mqtt_command(self):
        """超时重置MQTT指令"""
        with self.cmd_lock:
            self.mqtt_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print("[MQTT] 指令超时，已自动停止")
        if self.debug:
            print("[调试] MQTT指令已重置为停止状态")

    def get_obs(self):
        try:
            imu_data = self.imu.get_data()
            if self.debug and np.random.random() < 0.1:  # 10%概率打印IMU数据
                print(f"[调试] IMU数据: {imu_data}")

            dof_pos = self.hwi.get_present_positions(
                ignore=[
                    "left_antenna",
                    "right_antenna",
                ]
            )  # rad

            dof_vel = self.hwi.get_present_velocities(
                ignore=[
                    "left_antenna",
                    "right_antenna",
                ]
            )  # rad/s

            if dof_pos is None or dof_vel is None:
                print("[错误] 无法获取电机位置或速度数据")
                return None

            if len(dof_pos) != self.num_dofs:
                print(f"[错误] 电机位置数量不匹配: {len(dof_pos)} != {self.num_dofs}")
                return None

            if len(dof_vel) != self.num_dofs:
                print(f"[错误] 电机速度数量不匹配: {len(dof_vel)} != {self.num_dofs}")
                return None

            cmds = self.last_commands
            feet_contacts = self.feet_contacts.get()

            obs = np.concatenate(
                [
                    imu_data["gyro"],
                    imu_data["accelero"],
                    cmds,
                    dof_pos - self.init_pos,
                    dof_vel * 0.05,
                    self.last_action,
                    self.last_last_action,
                    self.last_last_last_action,
                    self.motor_targets,
                    feet_contacts,
                    self.imitation_phase,
                ]
            )

            return obs
        except Exception as e:
            print(f"[错误] 获取观测数据失败: {e}")
            return None

    def start(self):
        try:
            kps = [self.pid[0]] * 14
            kds = [self.pid[2]] * 14

            # 降低头部电机的比例系数
            kps[5:9] = [8, 8, 8, 8]

            self.hwi.set_kps(kps)
            self.hwi.set_kds(kds)
            self.hwi.turn_on()
            
            if self.debug:
                print(f"[调试] 电机已启动，PID参数: Kp={kps}, Kd={kds}")

            # 等待电机初始化完成
            time.sleep(2)
            
            # 验证电机连接状态（修改：检查HWI是否有is_connected属性）
            if hasattr(self.hwi, 'is_connected') and not self.hwi.is_connected():
                print("[警告] 电机可能未正确连接")
        except Exception as e:
            print(f"[错误] 电机启动失败: {e}")

    def get_phase_frequency_factor(self, x_velocity):
        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # 线性插值计算频率因子
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )
        return freq

    def run(self):
        i = 0
        try:
            print("开始运行控制循环")
            start_t = time.time()
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    # 获取手柄指令
                    try:
                        xbox_cmd, self.buttons, left_trigger, right_trigger = (
                            self.xbox_controller.get_last_command()
                        )
                        # 处理头控模式切换
                        if self.buttons.Y.triggered:
                            self.xbox_controller.head_control_mode = not self.xbox_controller.head_control_mode
                            mode = "头控模式" if self.xbox_controller.head_control_mode else "移动模式"
                            print(f"切换到{mode}")
                        
                        # 方向键调节相位频率偏移
                        if self.buttons.dpad_up.triggered:
                            self.phase_frequency_factor_offset += 0.05
                            print(f"相位频率因子偏移: {round(self.phase_frequency_factor_offset, 3)}")
                        if self.buttons.dpad_down.triggered:
                            self.phase_frequency_factor_offset -= 0.05
                            print(f"相位频率因子偏移: {round(self.phase_frequency_factor_offset, 3)}")
                        
                        # LB按钮控制相位频率因子
                        if self.buttons.LB.is_pressed:
                            self.phase_frequency_factor = 1.3
                        else:
                            self.phase_frequency_factor = 1.0
                        
                        # X按钮切换投影仪
                        if self.buttons.X.triggered and self.duck_config.projector:
                            self.projector.switch()
                        
                        # B按钮播放随机声音
                        if self.buttons.B.triggered and self.duck_config.speaker:
                            self.sounds.play_random_sound()
                        
                        # 天线控制
                        if self.duck_config.antennas:
                            self.antennas.set_position_left(right_trigger)
                            self.antennas.set_position_right(left_trigger)
                        
                        # A按钮暂停/继续
                        if self.buttons.A.triggered:
                            self.paused = not self.paused
                            print("暂停" if self.paused else "继续")
                        
                        # 合并MQTT指令和手柄指令（核心修改）
                        with self.cmd_lock:
                            mqtt_cmd = self.mqtt_command
                        # 优先使用MQTT指令（如果非零），否则使用手柄指令
                        self.last_commands = mqtt_cmd if any(mqtt_cmd) else xbox_cmd

                    except Exception as e:
                        if self.debug:
                            print(f"[调试] 获取手柄指令错误: {e}")

                if self.paused:
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    continue

                # 更新模仿相位
                self.imitation_i += 1 * (
                    self.phase_frequency_factor + self.phase_frequency_factor_offset
                )
                self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                self.imitation_phase = np.array([
                    np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                    np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi)
                ])

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("回放结束")
                        break

                # 推理动作
                action = self.policy.infer(obs)

                # 更新动作历史
                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # 计算电机目标位置
                self.motor_targets = self.init_pos + action * self.action_scale

                # 动作滤波
                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    if time.time() - start_t > 1:  # 等待滤波器稳定
                        self.motor_targets = self.action_filter.get_filtered_action()

                self.prev_motor_targets = self.motor_targets.copy()

                # 应用头部控制指令
                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                # 设置电机位置
                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )
                self.hwi.set_position_all(action_dict)

                i += 1

                # 控制循环频率管理
                took = time.time() - t
                if (1 / self.control_freq - took) < 0 and self.debug:
                    print(f"控制预算超出: {np.around(took - 1 / self.control_freq, 3)}秒")
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            if self.duck_config.antennas:
                self.antennas.stop()
            print("用户中断程序")

        if self.save_obs:
            pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
            print("观测数据已保存")
        print("程序终止，关闭电机")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="启用外部指令（手柄或键盘）",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="保存运行时的观测数据",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="回放之前保存的观测数据",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument("--debug", action="store_true", default=False, help="启用调试输出")

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("参数解析完成")
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
        debug=args.debug,
    )
    print("RLWalk实例化完成")
    rl_walk.run()