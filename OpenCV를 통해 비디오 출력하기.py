import cv2

# 카메라에 접근하기 위해 videocapture 객체를 생성
# 만약 비디오가 2개면 인자를 1로 주면됩니다.
cap = cv2.VideoCapture(0)

while(True):
  ret, img_color = cap.read()
  if ret == false:
    continue
  cv2.imshow("Color", img_color)
  if cv2.waitKey(1):
    break

cap.release()
cv2.destroyAllWindows()
  
  
