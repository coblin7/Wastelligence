import cv2

# 이미지 파일을 컬러로 읽어옵니다.
# 첫 번째 인자는 이미지 파일의 이름입니다.
# 두 번째 인자는 이미지를 읽을 때 사용되는 flag 입니다.
img_test = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

# 이미지 크기를 변환하여 출력합니다.
# i.e resize_img = cv2.resize(img_test, (1280, 720))

## cv2.imread(filename,[flags])
# flags의 값에 따라서 이미지를 다양하게 불러올 수 있습니다.
# cv2.IMREAD_COLOR는 투명도 정보를 포함하지 않은 이미지를 컬러로 불러옵니다.
# cv2.IMREAD_GRAYSCALE은 이미지를 회색조로 불러옵니다.
# cv2.IMREAD_UNCHANGED는 투명도 정보를 포함한 이미지를 컬러로 불러옵니다.

# 윈도우 창에 컬러 이미지가 보여지도록 식별자를 지정합니다.
# 첫 번째 인자는 윈도우 이미지에 대한 식별자 입니다.
cv2.namedWindow('Show Image')

## imshow 함수를 사용하여 지정한 윈도우에 이미지를 보여줍니다.
# imshow 함수만 사용하여도 named를 명시적으로 사용하지 않아도 자동으로 호출 됩니다.
# 첫 번째 인자는 윈도우 식별자입니다. 이미지가 보여지는 윈도우 창의 이름이 됩니다.
# 두 번째 인자는 윈도우에 보여줄 이미지에 대한 변수입니다.
cv2.imshow('Show Image', img_color)

# 지정한 숫자만큼 사용자의 키보드 입력을 기다립니다.
# 0으로 하게 되면 무한이 대기하게 됩니다.
cv2.waitKey(0)

# 프로그램이 종료 전 할당받았던 자원을 반환합니다.
cv2.destroyAllWindows()


## https://www.youtube.com/watch?v=w8iO9X5jcf8 참고영상
