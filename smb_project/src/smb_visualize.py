#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, math
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN

import rospy, rospkg, time

from std_msgs.msg import Int64, String
from sensor_msgs.msg import Image, LaserScan
from xycar_motor.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3

rospy.init_node('xycar')

CONTROL_TIME = 0.01
RATE = rospy.Rate(1 / CONTROL_TIME)
WIDTH, HEIGHT = 640, 480
ROI_ROW = 120


# IMAGE PROCESSING FOR CAM_DRIVING
class IMG_PROCESSING:
    def __init__(self):
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        
        self.image = np.empty(shape=[0])
        self.bridge = CvBridge()
        self.img_ready = False
        self.CAM_FPS = 30
        self.ROI_HEIGHT = HEIGHT - ROI_ROW 
        
    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.img_ready = True
        
    def is_image_ready(self):
        return self.img_ready and self.image.size == (WIDTH * HEIGHT * 3)
        
    def find_line(self):        
        img = self.image.copy()
        self.img_ready = False

        # Histogram Equalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # HE 적용
        hist_equalized = cv2.equalizeHist(gray)
        
        # GaussianBlur
        blur_gray = cv2.GaussianBlur(hist_equalized, (5, 5), 0)
        
        # Canny Edge Detection
        edge_img = cv2.Canny(np.uint8(blur_gray), 150, 500)
        roi_edge_img = edge_img[ROI_ROW:HEIGHT, 0:WIDTH]
        display_img = edge_img
        line_draw_img = roi_edge_img

        cv2.imshow('canny', edge_img)

        all_lines = cv2.HoughLinesP(roi_edge_img, 1, math.pi/180, 50, 30, 20)
        if all_lines is None:
            return [], [], [], [], 0

        left_x, right_x = [], []
        left_y, right_y = [], []
        horiz_line_num = 0

        for line in all_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < -0.2 and x2 < WIDTH / 2:
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)
                cv2.line(line_draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            elif slope > 0.2 and x1 > WIDTH / 2:
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)           
                cv2.line(line_draw_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            if -0.2 <= slope and slope <= 0.2:
                horiz_line_num += 1

        display_img[ROI_ROW:HEIGHT, 0:WIDTH] = line_draw_img
        cv2.imshow('find_line', display_img)
        
        return left_x, right_x, left_y, right_y, horiz_line_num
    
    def get_img(self):
        return self.image.copy()    


# LINE TRACKING BY USING CAMERA
# USAGE : TWO LINE TRACKING(MISSION 1), ONE LINE TRACKING(MISSION 2) 
class CAM_DRIVING:
    def __init__(self):
        self.img_proc = IMG_PROCESSING()
        self.prev_x_left = 0
        self.x_left = 0
        self.y_left = ROI_ROW
        self.x_right = WIDTH
        self.y_right = ROI_ROW
        self.x_midpoint = WIDTH // 2
        self.y_midpoint = ROI_ROW
        self.prev_x_left = 0
        self.prev_y_left = ROI_ROW
        self.prev_x_right = WIDTH
        self.prev_y_right = ROI_ROW
        self.prev_x_midpoint = WIDTH // 2
        self.prev_y_midpoint = ROI_ROW

    def find_midpoint_visualize(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()

        img = self.img_proc.get_img()
        display_img = img
        line_img = img.copy()[ROI_ROW:HEIGHT, 0:WIDTH]

        left_x, right_x, left_y, right_y, _ = self.img_proc.find_line()

        for i in range(0, len(left_x), 2):
            x1, x2 = left_x[i], left_x[i+1]
            y1, y2 = left_y[i], left_y[i+1]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for i in range(0, len(right_x), 2):
            x1, x2 = right_x[i], right_x[i+1]
            y1, y2 = right_y[i], right_y[i+1]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        display_img[ROI_ROW:HEIGHT, 0:WIDTH] = line_img
        cv2.imshow('find_midpoint', display_img)
        cv2.waitKey(1)

# MAIN LOOP
if __name__ == '__main__':
    cam_drive = CAM_DRIVING()
    
    while not rospy.is_shutdown():
        cam_drive.find_midpoint_visualize()
        
        RATE.sleep()