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
# 
import tensorflow as tf
# 텐서플로우 모듈을 불러옵니다.
import argparse
import sys
# 파이썬의 인터프리터를 제어방법을 제공하는 모듈을 불러옵니다.

# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
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
