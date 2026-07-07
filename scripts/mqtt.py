import warnings
# 在导入paho.mqtt之前设置警告过滤
warnings.filterwarnings(
    "ignore", 
    category=DeprecationWarning, 
    module="paho.mqtt"
)

import paho.mqtt.client as mqtt
import time
import random
import sys

# MQTT基本配置
BROKER = "peien.xyz"  # 公共MQTT代理
PORT = 1883                # 非加密端口
TOPIC = "2c:cf:67:56:d7:8c"       # 自定义主题

# === 订阅者部分 ===
def on_connect_sub(client, userdata, flags, rc):
    """订阅者连接回调"""
    if rc == 0:
        print("订阅者已连接到MQTT代理")
        client.subscribe(TOPIC)  # 订阅主题
    else:
        print(f"连接失败，返回码: {rc}")

def on_message(client, userdata, msg):
    """消息接收回调"""
    print(f"收到消息: {msg.payload.decode()} (主题: {msg.topic})")

def run_subscriber():
    """运行订阅者"""
    subscriber = mqtt.Client(
        client_id=f"sub-{random.randint(1, 1000)}",
        protocol=mqtt.MQTTv311  # 指定协议版本
    )
    subscriber.on_connect = on_connect_sub
    subscriber.on_message = on_message
    subscriber.connect(BROKER, PORT)
    subscriber.loop_forever()  # 持续监听消息

# === 发布者部分 ===
def on_connect_pub(client, userdata, flags, rc):
    """发布者连接回调"""
    if rc == 0:
        print("发布者已连接到MQTT代理")
    else:
        print(f"连接失败，返回码: {rc}")

def run_publisher():
    """运行发布者"""
    publisher = mqtt.Client(
        client_id=f"pub-{random.randint(1, 1000)}",
        protocol=mqtt.MQTTv311  # 指定协议版本
    )
    publisher.on_connect = on_connect_pub
    publisher.connect(BROKER, PORT)
    
    # 发布3条消息后退出
    for i in range(3):
        message = f"你好，这是第{i+1}条消息"
        publisher.publish(TOPIC, message)
        print(f"已发布: {message}")
        time.sleep(1)  # 间隔1秒
    
    publisher.disconnect()
    print("发布者已断开连接")

# === 主程序 ===
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else input("选择模式 (sub=订阅者, pub=发布者): ")
    
    if mode.lower() == "sub":
        run_subscriber()
    elif mode.lower() == "pub":
        run_publisher()
    else:
        print("请输入 sub 或 pub")