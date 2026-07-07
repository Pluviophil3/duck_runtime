from pypot.feetech import FeetechSTS3215IO

io = FeetechSTS3215IO("/dev/ttyACM0")

# 需要配置的舵机ID列表
servo_ids = [10,11,12,13,14,20,21,22,23,24,30,31,32,33]

# 基础配置参数（按需调整数值）
config = {
    'lock': 0,                #  解锁扭矩
    'acceleration': 0,      # 不限制最大加速度 
    'D_coefficient': 0        # 阻尼系数
}

# 逐个配置舵机
for sid in servo_ids:
    io.set_lock({sid: config['lock']})
    io.set_maximum_acceleration({sid: config['acceleration']})
    io.set_D_coefficient({sid: config['D_coefficient']})