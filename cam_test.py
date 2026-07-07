from picamzero import Camera
import cv2
import picamera2

print("Initializing camera ...")
try:
    print("Getting global camera info...")
    camera_info = picamera2.Picamera2.global_camera_info()
    print(f"Global camera info: {camera_info}")
    cam = Camera()
    print("Camera initialized successfully")
except Exception as e:
    print(f"Error initializing camera: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    im = cam.capture_array()
    im = cv2.resize(im, (512, 512))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    cv2.imwrite("/home/lxkj/Open_Duck_Mini_Runtime/aze.jpg", im)
    print("Image saved successfully")
except Exception as e:
    print(f"Error capturing or saving image: {e}")
    import traceback
    traceback.print_exc()