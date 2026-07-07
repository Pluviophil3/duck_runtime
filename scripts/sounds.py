import os
import random
import threading
from queue import Queue, Empty
import pygame
import time

class Sounds:
    def __init__(self, sound_directory="./", audio_device="hw:2,0", volume=1.0):
        self.sound_directory = sound_directory
        self.audio_device = audio_device
        self.sounds = {}
        self.queue = Queue()
        self.ok = True
        self.volume = volume  # 新增音量属性
        
        # 初始化 pygame 音频系统
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            print("Pygame audio initialized successfully")
        except pygame.error as e:
            print(f"Pygame audio initialization error: {e}")
            self.ok = False
            return

        if not os.path.exists(sound_directory):
            print(f"Sound directory not found: {sound_directory}")
            self.ok = False
            return

        try:
            for file in os.listdir(sound_directory):
                if file.endswith(".wav"):
                    sound_path = os.path.join(sound_directory, file)
                    if os.path.exists(sound_path):
                        self.sounds[file] = sound_path
                    else:
                        print(f"Sound file not found: {sound_path}")
        except Exception as e:
            print(f"Error listing sound files: {e}")
            self.ok = False

        if not self.sounds:
            print("No sound files found.")
            self.ok = False

        # Start playback worker thread
        self.worker_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.worker_thread.start()

    def _playback_worker(self):
        while True:
            try:
                sound_path = self.queue.get(timeout=1)
                try:
                    # 使用 pygame 播放音频
                    sound = pygame.mixer.Sound(sound_path)
                    sound.set_volume(self.volume)  # 设置音量
                    sound.play()
                    # 等待音频播放完成
                    time.sleep(sound.get_length())
                except pygame.error as e:
                    print(f"Pygame sound play error: {e}")
                    print(f"Attempting to convert {sound_path} to a compatible format...")
                    new_sound_path = sound_path.replace('.wav', '_converted.wav')
                    convert_result = subprocess.run([
                        "ffmpeg", "-i", sound_path, "-ac", "1", "-ar", "44100", "-b:a", "128k", new_sound_path
                    ], capture_output=True, text=True)

                    if convert_result.returncode == 0:
                        print(f"Conversion successful. Trying to play {new_sound_path}...")
                        try:
                            sound = pygame.mixer.Sound(new_sound_path)
                            sound.set_volume(self.volume)  # 设置音量
                            sound.play()
                            time.sleep(sound.get_length())
                        except pygame.error as e:
                            print(f"Still unable to play {new_sound_path}: {e}")
                    else:
                        print(f"Conversion failed: {convert_result.stderr}")
            except Empty:
                continue
            except Exception as e:
                print(f"Playback worker error: {e}")

    def play(self, sound_name):
        if not self.ok:
            return
        if sound_name in self.sounds:
            self.queue.put(self.sounds[sound_name])
        else:
            print(f"Sound not found: {sound_name}")

    def play_random_sound(self):
        if not self.ok or not self.sounds:
            return
        sound_name = random.choice(list(self.sounds.keys()))
        self.play(sound_name)

    def play_happy(self):
        self.play("sad-r2d2.wav")

# Example usage
if __name__ == "__main__":
    sound_directory = "/home/lxkj/Open_Duck_Mini_Runtime/mini_bdx_runtime/converted/"
    sound_player = Sounds(sound_directory, volume=0.9)  # 设置音量为 90%
    time.sleep(1)
    while True:
        sound_player.play_random_sound()
        time.sleep(3)