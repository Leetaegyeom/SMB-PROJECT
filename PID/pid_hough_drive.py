#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, math
import rospy, rospkg, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_motor.msg import xycar_motor

import signal
import sys
import os
import random

image = np.empty(shape=[0])

bridge = CvBridge()

motor = None

img_ready = False

CAM_FPS = 30
WIDTH, HEIGHT = 640, 480
ROI_ROW = 250 
ROI_HEIGHT = HEIGHT - ROI_ROW

def PID(input_data, kp, ki, kd):
    global start_time, end_time, prev_error, i_error
    end_time = time.time()
    dt = end_time - start_time
    start_time = end_time

    error = 320 - input_data
    derror = error - prev_error

    p_error = kp * error
    i_error = i_error + ki * error * dt
    d_error = kd * derror / dt

    output = p_error + i_error + d_error
    prev_error = error

    if output > 50:
        output = 50
    elif output < -50:
        output = -50

    return -output

def img_callback(data):
    global image, img_ready
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    img_ready = True

def drive(Angle, Speed):
    global motor
    motor_msg = xycar_motor()
    motor_msg.angle = Angle
    motor_msg.speed = Speed
    motor.publish(motor_msg)

def start():
    global image, img_ready
    global motor
    prev_x_left, prev_x_right = 0, WIDTH
    prev_x_midpoint = WIDTH // 2

    rospy.init_node('h_drive')

    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

    image_sub = rospy.Subscriber("/usb_cam/image_raw/",Image, img_callback)

    print("---------- Xycar ----------")

    while not image.size == (WIDTH * HEIGHT * 3):
        continue
    
    while not rospy.is_shutdown():
        
        while img_ready == False:
            continue

        img = image.copy()
        img_ready = False 

        # ==========================================

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
        edge_img = cv2.Canny(np.uint8(blur_gray), 30, 60)

        roi_edge_img = edge_img[ROI_ROW:HEIGHT, 0:WIDTH]

        if all_lines is None:
            continue

        for line in all_lines:
            x1, y1, x2, y2 = line[0]

        all_lines = cv2.HoughLinesP(roi_edge_img, 1, math.pi/180,50,30,20)

        if all_lines is None:
            continue

        left_x, right_x = [], []
        
        for line in all_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Prevent devide by 0

            if slope < -0.2 and x2 < WIDTH / 2:
                left_x.append(x1)
                left_x.append(x2)
            elif slope > 0.2 and x1 > WIDTH / 2:
                right_x.append(x1)
                right_x.append(x2)

        if left_x and right_x: # 2-Line driving
            x_left = sum(left_x) / len(left_x)
            x_right = sum(right_x) / len(right_x)
            x_midpoint = (x_left + x_right) // 2
        elif left_x: # 1-Line driving(Left Line)
            x_left = sum(left_x) / len(left_x)
            x_midpoint = x_left + (prev_x_midpoint - prev_x_left)
        elif right_x: # 1-Line driving(Right Line)
            x_right = sum(right_x) / len(right_x)
            x_midpoint = x_right + (prev_x_midpoint - prev_x_right)
        else: # 0-Line driving
            x_midpoint = prev_x_midpoint

        prev_x_left = x_left
        prev_x_right = x_right
        prev_x_midpoint = x_midpoint

        cv2.waitKey(1)

        # ==========================================

        angle = PID(x_midpoint, 0.6, 0.006, 0.001)

        speed = 5

        drive(angle, speed)

if __name__ == '__main__':

    i_error = 0.0
    prev_error = 0.0
    start_time = time.time()

    start()
