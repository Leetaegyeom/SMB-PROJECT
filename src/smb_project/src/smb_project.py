#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, math
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN

import rospy, rospkg, time

from sensor_msgs.msg import Image, LaserScan
from xycar_motor.msg import xycar_motor

CONTROL_TIME = 0.1
RATE = rospy.Rate(1 / CONTROL_TIME)
WIDTH, HEIGHT = 640, 320

P_GAIN = 0.6
I_GAIN = 0.006
D_GAIN = 0.001
SPEED = 5
THETA = 0
CENTER_X = 0

# SEND CONTROL MESSAGE TO XYCAR
class CONTROL:
    def __init__(self):
        self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        
        self.i_error = 0.0
        self.prev_error = 0.0
        self.ANGLE_LIMIT = 50
        
    def pid(self, input_data, kp, ki, kd, type):
        if type == "LINE_TRACKING":
            error = WIDTH // 2 - input_data
        elif type == "AVOID OBSTACLE":
            error = THETA - input_data
        elif type == "TUNNEL DRIVING":
            error = CENTER_X - input_data
            
        derror = error - self.prev_error

        p_error = kp * error
        self.i_error = self.i_error + ki * error * CONTROL_TIME
        d_error = kd * derror / CONTROL_TIME

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


# IMAGE PROCESSING FOR CAM_DRIVING
class IMG_PROCESSING:
    def __init__(self):
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        
        self.image = np.empty(shape=[0])
        self.bridge = CvBridge()
        self.img_ready = False
        self.CAM_FPS = 30
        self.ROI_ROW = 250
        self.ROI_HEIGHT = HEIGHT - self.ROI_ROW
        
    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.img_ready = True
        
    def is_image_ready(self):
        return self.img_ready and (not self.image.size == (WIDTH * HEIGHT * 3))
        
    def find_line(self):        
        img = self.image.copy()
        self.img_ready = False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_img = cv2.Canny(np.uint8(blur_gray), 30, 60)
        roi_edge_img = edge_img[self.ROI_ROW:HEIGHT, 0:WIDTH]

        all_lines = cv2.HoughLinesP(roi_edge_img, 1, math.pi/180, 50, 30, 20)
        if all_lines is None:
            return [], []

        left_x, right_x = [], []

        for line in all_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < -0.2 and x2 < WIDTH // 2:
                left_x.append(x1)
                left_x.append(x2)
                
            elif slope > 0.2 and x1 > WIDTH // 2:
                right_x.append(x1)
                right_x.append(x2)
                
        return left_x, right_x


# LINE TRACKING BY USING CAMERA
# USAGE : TWO LINE TRACKING(MISSION 1), ONE LINE TRACKING(MISSION 2) 
class CAM_DRIVING:
    def __init__(self):
        self.img_proc = IMG_PROCESSING()
        self.prev_x_left = 0
        self.prev_x_right = 0
        self.prev_x_midpoint = WIDTH // 2

    def find_midpoint(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()
        
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
            return None

        self.prev_x_left = x_left
        self.prev_x_right = x_right
        self.prev_x_midpoint = x_midpoint
        
        return x_midpoint


# LINE TRACKING BY USING LIDAR
# USAGE : OBSTACLE AVOIDANCE(MISSION 3), TUNNEL DRIVING(MISSION 4)
class LIDAR_DRIVING:
    def __init__(self):
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.lidar_points = None
        self.THETA2INPUT = 50 / 90 # theta : -90 ~ 90, input : -50 ~ 50
        self.LIDAR_ROI = [(0, 181), (540, 720)]

    def lidar_callback(self, data):
        self.lidar_points = data.ranges

    def preprocess_lidar_data(self):
        if self.lidar_points is None:
            return None

        ranges = np.array(self.lidar_points)
        valid_idx = (ranges > 0.1) & (ranges < 10.0)        # Adjust the min and max range as necessary
        points = np.column_stack((ranges[valid_idx] * np.cos(np.linspace(0, 2 * np.pi, len(ranges))[valid_idx]),
                                ranges[valid_idx] * np.sin(np.linspace(0, 2 * np.pi, len(ranges))[valid_idx])))

        return points

    def cluster(self, points):
        db = DBSCAN(eps=0.2, min_samples=3).fit(points)
        labels = db.labels_

        clusters = [points[labels == label] for label in set(labels) if label != -1]
        if not clusters:
            return None

        largest_cluster = max(clusters, key=len)
        center = np.mean(largest_cluster, axis=0)
        return center

    # FOR OBSTACLE AVOIDANCE(MISSION 3)
    def find_obstacle(self):
        points = self.preprocess_lidar_data()
        center = self.cluster(points)

        xycacr2obstacle_vec = np.asarray(center)
        xycacr2obstacle_theta = np.degrees(np.arctan2(xycacr2obstacle_vec[1], xycacr2obstacle_vec[0]))*self.THETA2INPUT
        return xycacr2obstacle_theta
     
    # FOR TUNNEL DRIVING(MISSION 4)
    def find_midpoint(self):
        points = self.preprocess_lidar_data()
        # 라이다 뒤집혀있으니까 이게 맞나 ? 위에서 좌표계변환할때 x y축 방향 알아야할듯
        left_points = points[points[:, 0] > 0]
        right_points = points[points[:, 0] < 0]

        right_wall_center = self.cluster(right_points)
        left_wall_center = self.cluster(left_points)

        total_center_x = (right_wall_center[0] + left_wall_center[0])/2
        return total_center_x



# MAIN LOOP
if __name__ == '__main__':
    rospy.init_node('xycar')
    xycar = CONTROL()
    cam_drive = CAM_DRIVING()
    lidar_drive = LIDAR_DRIVING()
    
    while not rospy.is_shutdown():
        cam_midpoint = cam_drive.find_midpoint()
        if cam_midpoint is None:
            midpoint = lidar_drive.find_midpoint()
        else:
            midpoint = cam_midpoint
        
        if midpoint is not None:
            angle = xycar.pid(midpoint, P_GAIN, I_GAIN, D_GAIN, "LINE_TRACKING")
            speed = SPEED  # Adjust speed as necessary
            xycar.drive(angle, speed)
        
        RATE.sleep()
