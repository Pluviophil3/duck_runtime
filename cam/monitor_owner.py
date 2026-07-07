import face_recognition
import time
import cv2
import subprocess
import numpy as np
import os

owner_image_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/owner_image.jpg"
owner_encoding_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/owner_encoding.npy"

# 加载‘主人’的人脸编码
if not os.path.exists(owner_encoding_path):
    print("未找到‘主人’人脸编码，先录入‘主人’！")
    exit(1)

owner_encoding = np.load(owner_encoding_path)


def monitor_owner():
    last_detected_time = 0

    print("开始监视‘主人’...")
    while True:
        # 使用 libcamera 捕捉图像并保存为临时文件
        temp_image_path = "/home/lxkj/Open_Duck_Mini_Runtime/cam/temp_monitor_image.jpg"
        cmd = [
            "libcamera-still",
            "-t", "1000",  # 等待 1 秒让相机自动调整曝光和白平衡
            "--nopreview",
            "-o", temp_image_path
        ]

        # 调用 libcamera 捕捉图片
        subprocess.run(cmd, check=True)

        # 读取图像
        img = cv2.imread(temp_image_path)

        # 转为RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 找到所有的人脸
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        if len(face_locations) > 0:
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces([owner_encoding], face_encoding)

                if True in matches:
                    current_time = time.time()
                    # 20秒间隔内识别到“主人”
                    if current_time - last_detected_time > 20:
                        print("看到主人！")
                        last_detected_time = current_time

        time.sleep(1)  # 每秒检查一次


if __name__ == "__main__":
    monitor_owner()
