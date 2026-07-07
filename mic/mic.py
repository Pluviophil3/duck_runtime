import sys
# 添加模块搜索路径
sys.path.append('/home/lxkj/Open_Duck_Mini_Runtime/scripts')
import pvporcupine
import pyaudio
import numpy as np
import time
from mini_bdx_runtime.sounds import Sounds
from v2_rl_walk_mujoco import RLWalk

# 唤醒词和命令词ppn模型路径
KEYWORD_PATHS = [
    '/home/lxkj/Open_Duck_Mini_Runtime/mic/HeyLingxi.ppn',  # 唤醒词：灵犀
    '/home/lxkj/Open_Duck_Mini_Runtime/mic/HeyLingxi.ppn',    # 前进
    '/home/lxkj/Open_Duck_Mini_Runtime/mic/HeyLingxi.ppn',     # 后退
    '/home/lxkj/Open_Duck_Mini_Runtime/mic/HeyLingxi.ppn',    # 停止
    '/home/lxkj/Open_Duck_Mini_Runtime/mic/HeyLingxi.ppn',   # 左转
    '/home/lxkj/Open_Duck_Mini_Runtime/mic/HeyLingxi.ppn',   # 右转
    '/home/lxkj/Open_Duck_Mini_Runtime/mic/HeyLingxio.ppn'  # 灵犀没事了
]
KEYWORDS = ['唤醒', '前进', '后退', '停止', '左转', '右转', '灵犀没事了']

PORCUPINE_ACCESS_KEY = '你的Porcupine密钥'  # 替换成你自己的Key

# 初始化声音类
sounds = Sounds(volume=1.0, sound_directory="../mini_bdx_runtime/assets/")

# 初始化运动控制类
rl_walk = RLWalk(
    "/home/lxkj/Open_Duck_Mini_Runtime/scripts/BEST_WALK_ONNX_2.onnx",
    cutoff_frequency=40,
)

# 定义运动控制函数
def move_forward():
    rl_walk.last_commands[0] = 0.15
    print("前进")
    sounds.play_random_sound()

def move_backward():
    rl_walk.last_commands[0] = -0.15
    print("后退")
    sounds.play_random_sound()

def stop():
    rl_walk.last_commands[0] = 0.0
    print("停止")
    sounds.play_random_sound()

def turn_left():
    rl_walk.last_commands[2] = 1.0
    print("左转")
    sounds.play_random_sound()

def turn_right():
    rl_walk.last_commands[2] = -1.0
    print("右转")
    sounds.play_random_sound()

if __name__ == '__main__':
    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keyword_paths=KEYWORD_PATHS
    )

    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("等待唤醒词“灵犀”...（Ctrl+C退出）")

    try:
        while True:
            # 等待唤醒词
            while True:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                keyword_index = porcupine.process(pcm)
                if keyword_index == 0:
                    sounds.play_random_sound()
                    print("灵犀在！请说指令...")
                    break

            # 等待命令词
            while True:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                cmd_index = porcupine.process(pcm)
                # 命令词 index: 1~5
                if cmd_index == 1:
                    move_forward()
                elif cmd_index == 2:
                    move_backward()
                elif cmd_index == 3:
                    stop()
                elif cmd_index == 4:
                    turn_left()
                elif cmd_index == 5:
                    turn_right()
                elif cmd_index == 6:  # 检测到“灵犀没事了”
                    sounds.play_random_sound()
                    print("好的，有需要再喊我。")
                    break
                # 支持超时退出命令状态，可以添加超时判断

            # 命令执行完，回到唤醒词等待

    except KeyboardInterrupt:
        print("程序终止")
    finally:
        porcupine.delete()
        audio_stream.close()
        pa.terminate()