# with pycamera

import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.resolution = (480,320)
camera.framerate = 32
# 카메라 변수에 설정했던 정보를 PiRGBArray 함수를 통해 메모리 배열을 구성합니다. (numpy.ndarray)
rawCapture = PiRGBArray(camera, size=(480,320))
sleep(0.1)

# continuous 함수를 통해 계속해서 캡쳐를 하며 이를 frame 변수에 저장 합니다.
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
  image = frame.array
  
  cv2.imshow('Frame', image)
  key=cv2.waitKey(1) & 0xFF
  rawCapture.truncate(0)
  if key == 27:
    break
    
