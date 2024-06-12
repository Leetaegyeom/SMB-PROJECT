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
ROI_ROW = 250 # ROI row
ROI_HEIGHT = HEIGHT - ROI_ROW
L_ROW = ROI_HEIGHT - 120 



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

    rospy.init_node('h_drive')
    r = rospy.Rate(5)

    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

    
    while not rospy.is_shutdown():
        drive(0, 4)
        r.sleep()

if __name__ == '__main__':

    i_error = 0.0
    prev_error = 0.0
    start_time = time.time()

    start()
