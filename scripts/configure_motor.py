from pypot.feetech import FeetechSTS3215IO
import argparse
import time

# 定义电机 ID 字典
motor_ids = {
    "left_hip_yaw": 20,
    "left_hip_roll": 21,
    "left_hip_pitch": 22,
    "left_knee": 23,
    "left_ankle": 24,
    "neck_pitch": 30,
    "head_pitch": 31,
    "head_yaw": 32,
    "head_roll": 33,
    "right_hip_yaw": 10,
    "right_hip_roll": 11,
    "right_hip_pitch": 12,
    "right_knee": 13,
    "right_ankle": 14,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    help="The port the motor is connected to. Default is /dev/ttyACM0. Use `ls /dev/tty* | grep usb` to find the port.",
    default="/dev/ttyACM0",
)
# 添加 --acceleration 参数，用于指定最大加速度值
parser.add_argument("--acceleration", help="The maximum acceleration to set to the motors.", type=int, required=True)
args = parser.parse_args()
io = FeetechSTS3215IO(args.port)

# 遍历每个电机 ID 并进行配置
for joint_name, motor_id in motor_ids.items():
    try:
        # 获取当前电机的配置信息
        kp = io.get_P_coefficient([motor_id])
        ki = io.get_I_coefficient([motor_id])
        kd = io.get_D_coefficient([motor_id])
        max_acceleration = io.get_maximum_acceleration([motor_id])
        acceleration = io.get_acceleration([motor_id])
        mode = io.get_mode([motor_id])

        # 打印当前电机的配置信息
        print(f"Configuring motor: {joint_name} (ID: {motor_id})")
        print(f"PID : {kp}, {ki}, {kd}")
        print(f"max_acceleration: {max_acceleration}")
        print(f"acceleration: {acceleration}")
        print(f"mode: {mode}")

        # 设置电机参数
        io.set_lock({motor_id: 0})
        io.set_mode({motor_id: 0})
        # 设置最大加速度为指定值
        io.set_maximum_acceleration({motor_id: args.acceleration})
        io.set_acceleration({motor_id: args.acceleration})
        io.set_P_coefficient({motor_id: 32})
        io.set_I_coefficient({motor_id: 0})
        io.set_D_coefficient({motor_id: 0})

        time.sleep(1)

        # 设置电机目标位置
        io.set_goal_position({motor_id: 0})

        time.sleep(1)

        # 打印配置后的电机信息
        print("===")
        print("Done configuring motor.")
        print(f"Motor id: {motor_id}")
        print(f"P coefficient : {io.get_P_coefficient([motor_id])}")
        print(f"I coefficient : {io.get_I_coefficient([motor_id])}")
        print(f"D coefficient : {io.get_D_coefficient([motor_id])}")
        print(f"acceleration: {io.get_acceleration([motor_id])}")
        print(f"max_acceleration: {io.get_maximum_acceleration([motor_id])}")
        print(f"mode: {io.get_mode([motor_id])}")
        print("===")
    except Exception as e:
        print(f"Error configuring motor {joint_name} (ID: {motor_id}): {e}")