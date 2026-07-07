import cv2
import face_recognition
import numpy as np
import subprocess
import os
import time
import threading  # 导入多线程模块
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI

# 保存人脸图像和编码
owner_image_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/owner_image.jpg"
owner_encoding_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/owner_encoding.npy"
temp_image_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/temp_owner_image.jpg"

def play_sound(sound_path):
    """单独的函数：播放声音（用于多线程）"""
    import pygame
    pygame.mixer.init()
    if os.path.exists(sound_path):
        try:
            sound = pygame.mixer.Sound(sound_path)
            sound.set_volume(1.0)
            sound.play()
            # 等待声音播放完成
            pygame.time.wait(int(sound.get_length() * 1000))
        except Exception as e:
            print(f"播放声音出错: {e}")
    else:
        print(f"声音文件不存在: {sound_path}")

def shake_head(hwi):
    """单独的函数：控制摇头（用于多线程）"""
    head_yaw_range = [-60, 60]  # 摇头角度范围（度）
    num_shakes = 3  # 摇头次数
    shake_interval = 0.5  # 每次摇头间隔时间（秒）

    for _ in range(num_shakes):
        # 向左摇头
        head_yaw_deg = head_yaw_range[0]
        head_yaw_pos_rad = np.deg2rad(head_yaw_deg)
        hwi.set_position("head_yaw", head_yaw_pos_rad)
        time.sleep(shake_interval)

        # 向右摇头
        head_yaw_deg = head_yaw_range[1]
        head_yaw_pos_rad = np.deg2rad(head_yaw_deg)
        hwi.set_position("head_yaw", head_yaw_pos_rad)
        time.sleep(shake_interval)

    # 回到初始位置
    hwi.set_position("head_yaw", 0)

def capture_owner_face():
    while True:
        # 拍摄照片
        cmd = [
            "libcamera-still",
            "-t", "3000",
            "--nopreview",
            "-o", temp_image_path
        ]
        subprocess.run(cmd, check=True)
        print(f"照片已保存到: {temp_image_path}")
        
        # 读取并处理图片
        img = cv2.imread(temp_image_path)
        if img is None:
            print(f"图片读取失败: {temp_image_path}")
            continue
        
        print(f"原始图片shape: {img.shape}")
        scale = 800.0 / max(img.shape[0], img.shape[1])
        if scale < 1.0:
            new_w, new_h = int(img.shape[1] * scale), int(img.shape[0] * scale)
            img_small = cv2.resize(img, (new_w, new_h))
        else:
            img_small = img

        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb, model="hog")
        print("检测到的人脸位置:", face_locations)

        if not face_locations:
            print("未检测到人脸，重试...")
            time.sleep(1)
            continue
        
        # 选择最大人脸并保存
        largest_face_idx = np.argmax([(b-t)*(r-l) for t, r, b, l in face_locations])
        largest_face_encoding = face_recognition.face_encodings(img_rgb, [face_locations[largest_face_idx]])[0]
        top, right, bottom, left = face_locations[largest_face_idx]
        
        owner_face_img = img_rgb[top:bottom, left:right]
        cv2.imwrite(owner_image_path, cv2.cvtColor(owner_face_img, cv2.COLOR_RGB2BGR))
        np.save(owner_encoding_path, largest_face_encoding)
        print("‘主人’已录入！")

        # 初始化硬件和声音路径
        duck_config = DuckConfig()
        sound_path = os.path.join("../mini_bdx_runtime/assets/", "i_see_you.wav")
        hwi = HWI(duck_config)
        kps = [8] * 14
        kds = [0] * 14
        hwi.set_kps(kps)
        hwi.set_kds(kds)
        hwi.turn_on()

        # 创建并启动线程（同时播放声音和摇头）
        threads = []
        # 声音线程
        if duck_config.speaker:
            sound_thread = threading.Thread(target=play_sound, args=(sound_path,))
            threads.append(sound_thread)
        # 摇头线程
        head_thread = threading.Thread(target=shake_head, args=(hwi,))
        threads.append(head_thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        break  # 完成后退出循环

if __name__ == "__main__":
    capture_owner_face()