from picamera import PiCamera
from time import sleep

# camera 변수에 파이 카메라 클래스의 인스턴스를 만들어줍니다.
camera = PiCamera()

# 이미지를 찍을 해상도를 정합니다.
# 해당 해상도는 파이카메라의 최상위 해상도입니다.
camera.resolution = (2592, 1944)

#카메라의 프레임을 정해줍니다.
camera.framerate = 15

# 카메라의 조도나 초점을 맞추기 위해 대기 시간(sleep)을 정해줍니다.
camera.start_preview()
sleep(5)
# 카메라 동작과 함께 저장할 경로와 파일의 이름을 정해줍니다.
camera.capture('/home/pi/image.jpg')
camera.stop_preview()

# 참고문헌 https://neosarchizo.gitbooks.io/raspberrypiforsejonguniv/content/chapter4.html 
