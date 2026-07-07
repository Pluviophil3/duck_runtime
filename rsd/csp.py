import RPi.GPIO as GPIO  # 修正导入语句的错误后缀
import time  # 补全正确的导入语句

# ---------------------- 引脚定义 ----------------------
# HC-SR501人体红外传感器输出引脚（BCM编码）
SR501_OUT = 17
# L298N电机驱动板引脚定义（左电机：IN1/IN2，右电机：IN3/IN4）
IN1 = 22
IN2 = 23
IN3 = 24
IN4 = 25

# ---------------------- 初始化GPIO ----------------------
def init_gpio():
    """初始化GPIO引脚配置"""
    GPIO.setmode(GPIO.BCM)  # 使用树莓派BCM引脚编码规则
    GPIO.setwarnings(False)  # 关闭GPIO引脚复用警告

    # 配置HC-SR501传感器引脚为输入模式
    GPIO.setup(SR501_OUT, GPIO.IN)

    # 配置电机驱动引脚为输出模式，并初始化为低电平（电机停止）
    motor_pins = [IN1, IN2, IN3, IN4]
    for pin in motor_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

# ---------------------- 电机控制函数 ----------------------
def car_forward():
    """控制小车前进"""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def car_stop():
    """控制小车停止"""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def car_left():
    """控制小车原地左转（左电机反转，右电机正转）"""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def car_right():
    """控制小车原地右转（左电机正转，右电机反转）"""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

# ---------------------- 主循环：检测与避让逻辑 ----------------------
def main():
    """程序主函数：实现红外检测与小车避让"""
    init_gpio()  # 初始化GPIO
    print("HC-SR501避让程序启动，按Ctrl+C退出...")

    try:
        while True:
            # 读取HC-SR501传感器的输出状态
            sr501_status = GPIO.input(SR501_OUT)

            if sr501_status == GPIO.HIGH:
                # 检测到移动物体，执行避让动作
                print("检测到移动物体，执行避让...")
                car_stop()       # 小车停止
                time.sleep(0.5)  # 停止0.5秒
                car_left()       # 原地左转2秒（可替换为car_right()实现右转）
                time.sleep(2)
                car_forward()    # 继续前进
            else:
                # 未检测到物体，小车正常前进
                car_forward()

            time.sleep(0.1)  # 短延时，降低CPU占用率

    except KeyboardInterrupt:
        # 捕获Ctrl+C中断信号，安全退出程序
        print("\n程序退出，清理GPIO资源...")
        car_stop()          # 确保小车停止
        GPIO.cleanup()      # 清理GPIO配置

# 程序入口
if __name__ == "__main__":
    main()