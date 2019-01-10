######## Picamera Object Detection Using Tensorflow Classifier #########
# 텐서플로우 사물분류를 사용한 라즈베리파이 카메라 사물인식
# Author: Evan Juras
# Date: 4/15/18
# Description: 
#
# This program uses a TensorFlow classifier to perform object detection.
# 이 프로그램은 텐서플로우를 사용하여 사물인식을 수행합니다.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# 라즈베리파이 카메라를 통하여 사물인식을 수행하여 분류합니다.
# It draws boxes and scores around the objects of interest in each frame from
# 각 프레임 마다 사물들에 대해 박스와 점수를 보여줍니다.
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# 라즈베리파이 카메라 뿐만 아니라 usb 웹캠도 사용할 수 있습니다.
# when executing this script from the terminal.
# 터미널에서 이 스크립트를 실행할 때 동작합니다.
## Some of the code is copied from Google's example at
# 코드의 일부분은 구글 예제에서 복사하였습니다.
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py
## but I changed it to make it more understandable to me.
# 하지만 좀 더 이해를 돕기위해 수정하였습니다.

# Import packages
# 파이썬 패키지를 불러옵니다.
import os
# 운영체제를 제어하는 모듈을 불러옵니다. (파일과 폴더의 생성 및 복사 등)
import cv2
# OpenCV 영상처리 라이브러리 모듈을 불러옵니다. (이미지나 비디오를 읽고 보고 저장 등을 할 수 있습니다.)

import numpy as np
# 데이터 분석을 위한 수학연산(행렬, 벡터 연산등의 기능을 제공) 모듈을 불러옵니다.
# C언어로 구현된 파이썬 라이브러리 numpy는 고성능의 수치계산을 위해 만들어졌습니다.

from picamera.array import PiRGBArray
# (1) 파이카메라 모듈을 불러옵니다. 카메라 출력물로부터 n차원 numpy 배열을 구성하는 클래스를 제공합니다.
# 이 모듈은 자동으로 import되지 않기 때문에, 명시적으로 import 해주어야 합니다.
# 카메라의 데이터를 통해 배열을 구성하기 위해서 명시적으로 위 모듈을 불러와 주어야 한다.
# (2) picamera.array 모듈에서 PiRGBArray 함수를 불러옵니다.
# 위 함수는 RGB 캡처를 통해 3차원 RGB 배열(행,열,색상)을 생성합니다.
## https://picamera.readthedocs.io/en/release-1.10/api_array.html
## 파이썬은 import를 통해 전체 모듈을 가져오는 것과 from (모듈) import (변수or함수) 를 통해 특정 모듈만 불러올 수 있습니다.

from picamera import PiCamera
# 파이카메라 모듈을 불러옵니다.
import tensorflow as tf
# 텐서플로우 모듈을 불러옵니다.

import argparse
# 명령행 인터페이스에서 명령행 인자를 받아 실행되는 프로그램일 때 파싱 작업이 필요하다.
# 파싱 : 문장을 이루고 있는 구성 성분을 분해하고 위계 관계를 분석해 문장의 구조를 결정한다. -> 데이터를 조립해 원하는 데이터를 빼내는 것/ 문장 문석 및 문법 관계 해석
# 명령문의 작성한 코드의 변수나 함수를 구분하게 된다.
# argparse 모듈을 사용하면 매우 간단하게 명령행 인자를 파싱할 수 있다. (문법 안내 및 헬프 메시지 자동 생성)

import sys
# 파이썬의 인터프리터를 제어방법을 제공하는 모듈을 불러옵니다.

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate
# 해상도의 크기를 결정할 변수를 선언


# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
# usb카메라인지 파이카메라인지 카메라의 타입을 정한다.
parser = argparse.ArgumentParser()
# argparse 클래스를 불러옴으로써 parser 인스턴스(객체)를 생성 
# ArgumentParser 객체는 명령행을 파이썬 데이터형으로 파싱하는데 필요한 모든 정보를 담고 있다.
# ArgumentParser() = 객체 생성하기

parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
# parser는 인터프리터나 컴파일러의 구성요소 중 하나로 입력 token에 내제된 자료 구조를 빌드하고 문법을 검사한다.
# parser에게 add_argument 함수를 통해 명령문에 --usbcam 이라는 인자를 명령행에서 만날 때 수행하는 action을 통해 동작을 수행합니다.
# help는 도움말과 같음, store는 인자 값을 저장합니다. -> store_ ... -> store_true 는 --usbcam=true 가 되는 것입니다.
# parser 인스턴스에 --usbcam이라는 문법(인자)을 추가한 것입니다.
# add_argument() = 인자 추가하기

args = parser.parse_args()
# parse_args() 메소드를 통해 인자를 파싱합니다. 
# 명령행을 검사하고 인자를 적절한 형태로 변환하여 적절한 액션을 호출하게 됩니다.
# parse_args() = 인자 파싱하기

if args.usbcam:
    camera_type = 'usb'
# 명령문에 usbcam 인자가 있다면, camera_type변수는 usb 값을 가지게 됩니다.

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')
# 작업을 수행할 폴더가 object_detction 폴더이므로 이동할 필요가 있다. (하지만 명령문에서 이미 폴더 경로를 정해주고 이 소스코드를 실행하였기 때문에 ...이다.
# sys.path 만 수행할 경우 파이썬 라이브러리들이 설치되어있는 디렉토리 경로를 보여준다.
# 원하는 경로를 설정하기 위해서 append를 사용하여 경로를 설정해줄 수 있다. (예: sys.path.append("C:/.../...")

# Import utilites
from utils import label_map_util
# utilites는 기본적인 영상처리 작업을 수행할 수 있는 모듈입니다.
# label_map_util에 대한 정보 찾을 수 없음.
from utils import visualization_utils as vis_util
# visualization_utils는 Python에서 기계학습을 위한 시각화 유틸리티 기능입니다. 

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# object detectio 모듈에 사용될 모델의 이름이다.
# ssd는 single shot detector를 뜻하며, lite버전으로 mobilenet 모바일에서 사용되는 coco모델이다.
# ssd(Single Shot Detector), Yolo, RCN 들은 detector중 하나이다.

# Grab path to current working directory
# 현재의 작업 디렉토리의 경로를 파악합니다.
CWD_PATH = os.getcwd()
# CWD_PATH변수에 os.getcwd()는 현재 자신의 디렉토리 위치를 리턴합니다.

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
# object detection을 위해서 경로에 있는 frozen_detection_graph.pb 파일안에 있는 모델을 사용합니다.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
# label의 맵 파일의 경로를 지정합니다.
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')
# os.path.join()은 경로를 병합하여 새로운 경로를 생성합니다. PATH_TO_LABELS 변수에 새로운 경로를 삽입합니다.


# Number of classes the object detector can identify
# object detector가 식별할 수 있는 클래스의 수를 나타냅니다.
NUM_CLASSES = 90

## Load the label map.
# 레이블 맵을 불러와 현재까지 분류한 내용을 맵핑해줍니다.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
# 텐서플로우를 통해 그래프안에 데이터를 넣어 영상처럼 그려줍니다.
# 뉴런네트워크, 딥러닝 초기강좌
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# opencv구역
# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
# 카메라에 이미지가 들어오면 박스와 텍스트를 입히는 GUI와 관련된 내용이다.
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = frame1.array
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
elif camera_type == 'usb':
    # Initialize USB webcam feed
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()

cv2.destroyAllWindows()
