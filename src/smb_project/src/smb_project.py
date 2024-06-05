#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, math
from cv_bridge import CvBridge
import rospy, rospkg, time
from sensor_msgs.msg import Image, LaserScan
from xycar_motor.msg import xycar_motor


# IMAGE PROCESSING FOR CAM_DRIVING
class IMG_PROCESSING:
    def __init__(self):
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        
        self.image = np.empty(shape=[0])
        self.bridge = CvBridge()
        self.img_ready = False
        self.WIDTH, self.HEIGHT = 640, 480
        self.CAM_FPS = 30
        self.ROI_ROW = 250
        self.ROI_HEIGHT = self.HEIGHT - self.ROI_ROW
        
    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.img_ready = True
        
    def is_image_ready(self):
        return self.img_ready and (not self.image.size == (self.WIDTH * self.HEIGHT * 3))
        
    def find_line(self):
        img = self.image.copy()
        self.img_ready = False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
        edge_img = cv2.Canny(np.uint8(blur_gray), 30, 60)
        roi_edge_img = edge_img[self.ROI_ROW:self.HEIGHT, 0:self.WIDTH]

        all_lines = cv2.HoughLinesP(roi_edge_img, 1, math.pi/180,50,30,20)

        left_x, right_x = [], []

        for line in all_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < -0.2 and x2 < self.WIDTH // 2:
                left_x.append(x1)
                left_x.append(x2)
                
            elif slope > 0.2 and x1 > self.WIDTH // 2:
                right_x.append(x1)
                right_x.append(x2)
                
        return left_x, right_x


# LINE TRACKING BY USING CAMERA
# USAGE : TWO LINE TRACKING(MISSION 1), ONE LINE TRACKING(MISSION 2) 
class CAM_DRIVING:
    def __init__(self):
        rospy.init_node('cam_driving')
        self.img_proc = IMG_PROCESSING()
        self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        self.WIDTH, self.HEIGHT = 640, 480

        self.i_error = 0.0
        self.prev_error = 0.0

        self.P_GAIN = 0.6
        self.I_GAIN = 0.006
        self.D_GAIN = 0.001
        self.SPEED = 5

        self.ANGLE_LIMIT = 50

        self.dt = 0.1
        self.rate = rospy.Rate(1/self.dt)

    def pid(self, input_data, kp, ki, kd):
        error = self.WIDTH // 2 - input_data
        derror = error - self.prev_error

        p_error = kp * error
        self.i_error = self.i_error + ki * error * self.dt
        d_error = kd * derror / self.dt

        output = p_error + self.i_error + d_error
        self.prev_error = error

        if output > self.ANGLE_LIMIT:
            output = self.ANGLE_LIMIT
        elif output < -self.ANGLE_LIMIT:
            output = -self.ANGLE_LIMIT

        return -output

    def drive(self, Angle, Speed):
        motor_msg = xycar_motor()
        motor_msg.angle = Angle
        motor_msg.speed = Speed
        self.motor_pub.publish(motor_msg)

    def line_tracking(self):
        while not rospy.is_shutdown():
            if not self.img_proc.img_ready():
                continue

            left_x, right_x = self.img_proc.find_line()
            
            if left_x and right_x:
                x_left = sum(left_x) / len(left_x)
                x_right = sum(right_x) / len(right_x)
                x_midpoint = (x_left + x_right) // 2
                
            elif left_x:
                x_left = sum(left_x) / len(left_x)
                x_midpoint = x_left + (self.prev_x_midpoint - self.prev_x_left)
                
            elif right_x:
                x_right = sum(right_x) / len(right_x)
                x_midpoint = x_right + (self.prev_x_midpoint - self.prev_x_right)
                
            else:
                continue

            self.prev_x_left = x_left
            self.prev_x_right = x_right
            self.prev_x_midpoint = x_midpoint

            self.drive(self.pid(x_midpoint, self.P_GAIN, self.I_GAIN, self.D_GAIN), self.SPEED)
            
            self.rate.sleep()


# WALL TRACKING BY USING LIDAR
# USAGE : OBSTACLE AVOIDANCE(MISSION 3), TUNNEL DRIVING(MISSION 4)
class LIDAR_DRIVING:
    def __init__(self):
        rospy.init_node('lidar_driving')
        self.lidar_points = None
        lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)

    def lidar_callback(self, data):
        self.lidar_points = data.ranges


## MAIN CODE EX
# if __name__ == '__main__':
#     cam_driving = CAM_DRIVING()
#     cam_driving.start()
