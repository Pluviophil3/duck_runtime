import cv2
import face_recognition
import numpy as np
import subprocess
import os
import time
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.duck_config import DuckConfig
from mini_bdx_runtime.rustypot_position_hwi import HWI

# 保存人脸图像和编码
owner_image_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/owner_image.jpg"
owner_encoding_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/owner_encoding.npy"
temp_image_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/temp_owner_image.jpg"

def capture_owner_face():
    # 不断循环拍照，直到检测到人脸为止
    while True:
        # 使用 libcamera 捕捉图像并保存为临时文件
        cmd = [
            "libcamera-still",
            "-t", "3000",  # 等待 3 秒让相机自动调整曝光和白平衡
            "--nopreview",
            "-o", temp_image_path
        ]
        subprocess.run(cmd, check=True)
        print(f"照片已保存到: {temp_image_path}")
        
        # 读取图片
        img = cv2.imread(temp_image_path)
        if img is None:
            print(f"图片读取失败，请检查路径和文件：{temp_image_path}")
            continue
        
        print(f"原始图片shape: {img.shape}")
        
        # 缩小图片做检测
        scale = 800.0 / max(img.shape[0], img.shape[1])
        if scale < 1.0:
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            img_small = cv2.resize(img, (new_w, new_h))
            print(f"缩放后图片shape: {img_small.shape}")
        else:
            img_small = img

        # 转为RGB格式
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        # 找到所有的人脸
        face_locations = face_recognition.face_locations(img_rgb, model="hog")
        print("检测到的人脸位置（缩小图）:", face_locations)

        if len(face_locations) == 0:
            print("没有检测到人脸，正在重新拍照...")
            time.sleep(1)  # 等待 1 秒钟后再试
            continue  # 继续下次拍照
        
        # 选择最大的人脸区域
        largest_face_index = np.argmax([l[2] * l[3] for l in face_locations])  # 最大人脸的索引
        largest_face_encoding = face_recognition.face_encodings(img_rgb, [face_locations[largest_face_index]])[0]
        largest_face_location = face_locations[largest_face_index]
        
        print(f"最大人脸坐标: {largest_face_location}")
        
        # 截取最大的人脸区域并保存
        top, right, bottom, left = largest_face_location
        owner_face_img = img_rgb[top:bottom, left:right]
        cv2.imwrite(owner_image_path, cv2.cvtColor(owner_face_img, cv2.COLOR_RGB2BGR))

        # 保存人脸编码
        np.save(owner_encoding_path, largest_face_encoding)
        print("‘主人’已录入！")

       # # 发出提示音(随机播放）
       # duck_config = DuckConfig()
       #  if duck_config.speaker:
       #     sounds = Sounds(volume=1.0, sound_directory="../mini_bdx_runtime/assets/")
       #     sounds.play_random_sound()
            
            
    # 发出提示音（固定播放 i_see_you.wav）
        duck_config = DuckConfig()
        if duck_config.speaker:
            import pygame
            # 初始化pygame音频
            pygame.mixer.init()
            # 构建声音文件路径
            sound_path = os.path.join("../mini_bdx_runtime/assets/", "i_see_you.wav")
            # 检查文件是否存在
            if os.path.exists(sound_path):
                try:
                    sound = pygame.mixer.Sound(sound_path)
                    sound.set_volume(1.0)  # 设置音量
                    sound.play()
                    # 等待声音播放完成（避免程序提前结束）
                    pygame.time.wait(int(sound.get_length() * 1000))
                except Exception as e:
                    print(f"播放声音时出错: {e}")
            else:
                print(f"错误：未找到声音文件 {sound_path}")




        # 左右开心地摇头
        hwi = HWI(duck_config)
        kps = [8] * 14
        kds = [0] * 14
        hwi.set_kps(kps)
        hwi.set_kds(kds)
        hwi.turn_on()

        head_yaw_range = [-60, 60]  # 假设摇头的角度范围
        num_shakes = 3  # 摇头的次数
        shake_interval = 0.5  # 每次摇头的间隔时间

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
        head_yaw_deg = 0
        head_yaw_pos_rad = np.deg2rad(head_yaw_deg)
        hwi.set_position("head_yaw", head_yaw_pos_rad)

        break  # 找到人脸后跳出循环

if __name__ == "__main__":
    capture_owner_face()