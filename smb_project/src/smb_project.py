#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, math
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN

import rospy, rospkg, time

from std_msgs.msg import Int64, String
from sensor_msgs.msg import Image, LaserScan, PointCloud2, PointField
from xycar_motor.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

rospy.init_node('xycar')


CONTROL_TIME = 0.1
RATE = rospy.Rate(1 / CONTROL_TIME)
WIDTH, HEIGHT = 640, 480
ROI_ROW = 280
ROI_OFFSET = 50

P_GAIN_CAM = 0.6
I_GAIN_CAM = 0.006
D_GAIN_CAM = 0.001

P_GAIN_TUNNEL = 500
I_GAIN_TUNNEL = 0.0
D_GAIN_TUNNEL = 0.0

P_GAIN_OBS = 1000
I_GAIN_OBS = 0.006
D_GAIN_OBS = 0.001

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
        
        self.kp = 0
        self.ki = 0
        self.kd = 0
        
    def pid(self, input_data, type):
        if type == "LINE_TRACKING":
            error = WIDTH // 2 - input_data
            self.kp = P_GAIN_CAM
            self.ki = I_GAIN_CAM
            self.kd = D_GAIN_CAM

        elif type == "AVOID OBSTACLE":
            error = THETA - input_data
            self.kp = P_GAIN_OBS
            self.ki = I_GAIN_OBS
            self.kd = D_GAIN_OBS

        elif type == "TUNNEL DRIVING":
            error = CENTER_X - input_data
            self.kp = P_GAIN_TUNNEL
            self.ki = I_GAIN_TUNNEL
            self.kd = D_GAIN_TUNNEL
            
        derror = error - self.prev_error

        p_error = self.kp * error
        self.i_error = self.i_error + self.ki * error * CONTROL_TIME
        d_error = self.kd * derror / CONTROL_TIME

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
        self.ROI_HEIGHT = HEIGHT - ROI_ROW - ROI_OFFSET
        
    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.img_ready = True
        
    def is_image_ready(self):
        return self.img_ready and self.image.size == (WIDTH * HEIGHT * 3)
        
    def find_line(self):        
        img = self.image.copy()
        self.img_ready = False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_img = cv2.Canny(np.uint8(blur_gray), 30, 60)
        roi_edge_img = edge_img[ROI_ROW:HEIGHT-ROI_OFFSET, 0:WIDTH]

        all_lines = cv2.HoughLinesP(roi_edge_img, 1, math.pi/180, 50, 30, 20)
        if all_lines is None:
            return [], [], [], [], 0

        left_x, right_x = [], []
        left_y, right_y = [], []
        horiz_line_num = 0

        for line in all_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < 0.1 and x2 < WIDTH // 2:
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)
                
            elif slope > 0.1 and x1 > WIDTH // 2:
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)                
            
            if -0.2 <= slope and slope <= 0.2:
                horiz_line_num += 1
        
        return left_x, right_x, left_y, right_y, horiz_line_num

    def get_img(self):
        return self.image.copy()
    
    
# AR TAG DETECTION & IDENTIFICATION
class AR_TAG:
    def __init__(self):
        self.arData = {"ID":[], "DZ":[]}

        rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.AR_callback)

    def AR_callback(self, data):
        self.arData = {"ID":[], "DZ":[]}

        for i in data.markers:
            self.arData["ID"].append(i.id)
            self.arData["DZ"].append(i.pose.pose.position.z)

    def AR_detect(self):
        min_ID = None
        min_distance = float('inf')

        if len(self.arData["ID"]) == 0:
            return None, min_distance
        
        for idx, distance in enumerate(self.arData["DZ"]):
            if distance < min_distance:
                min_distance = distance
                min_ID = self.arData["ID"][idx]


# TRAFFIC LIGHT 
class TRAFFIC_LIGHT:
    def __init__(self):
        self.single_color = None
        self.prev_single_color = None
        self.right_color = None
        self.left_color = None
        self.time_count = None

        rospy.Subscriber("/Single_color", String, self.single_callback)
        rospy.Subscriber("/Right_color", String, self.right_callback)
        rospy.Subscriber("/Left_color", String, self.left_callback)
        rospy.Subscriber("/time_count", Int64, self.time_callback)

    def single_callback(self, data):
        self.prev_single_color = self.single_color
        self.single_color = data.data

    def right_callback(self, data):
        self.right_color = data.data

    def left_callback(self, data):
        self.left_color = data.data

    def time_callback(self, data):
        self.time_count = data.data

    def traffic_single(self):
        gostop = None

        if self.single_color is None:
            return None

        if self.single_color == 'G':
            gostop = 'go'
        
        elif self.single_color == 'Y':
            if self.prev_single_color == 'G':
                gostop = 'go'
            
            elif self.prev_single_color == 'Y':
                gostop = 'stop'
            
        elif self.single_color == 'R':
            gostop =  'stop'

        return gostop
    
    def traffic_crossroad(self):
        direction = None
        gostop = None

        if self.left_color is None or self.right_color is None or self.time_count is None:
            return None, None

        if 3 <= self.time_count:
            gostop = 'go'
        else:
            gostop = 'stop'

        if self.left_color == 'R':
            direction = 'right'
        else:
            direction = 'left'

        return gostop, direction
        

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

    def find_midpoint(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()
        
        left_x, right_x, _, _, _ = self.img_proc.find_line()
        
        if left_x and right_x:
            self.x_left = sum(left_x) / len(left_x)
            self.x_right = sum(right_x) / len(right_x)
            self.x_midpoint = (self.x_left + self.x_right) // 2
            
        elif left_x:
            self.x_left = sum(left_x) / len(left_x)
            self.x_midpoint = self.x_left + (self.prev_x_midpoint - self.prev_x_left)
            
        elif right_x:
            self.x_right = sum(right_x) / len(right_x)
            self.x_midpoint = self.x_right + (self.prev_x_midpoint - self.prev_x_right)
            
        else:
            return None

        self.prev_x_left = self.x_left
        self.prev_x_right = self.x_right
        self.prev_x_midpoint = self.x_midpoint
        
        return self.x_midpoint
    
    def find_midpoint_visualize(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()
            
        img = self.img_proc.get_img()
        display_img = img
        line_img = img.copy()[ROI_ROW:HEIGHT-ROI_OFFSET, 0:WIDTH]
        
        left_x, right_x, left_y, right_y, _ = self.img_proc.find_line()
        
        for i in range(0, len(left_x), 2):
            x1, x2 = left_x[i], left_x[i+1]
            y1, y2 = left_y[i], left_y[i+1]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        for i in range(0, len(right_x), 2):
            x1, x2 = right_x[i], right_x[i+1]
            y1, y2 = right_y[i], right_y[i+1]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
        if left_x and right_x:
            self.x_left = sum(left_x) / len(left_x)
            self.y_left = sum(left_y) / len(left_y)
            self.x_right = sum(right_x) / len(right_x)
            self.y_right = sum(right_y) / len(right_y)            
            self.x_midpoint = (self.x_left + self.x_right) // 2
            self.y_midpoint = (self.y_left + self.y_right) // 2
            cv2.rectangle(line_img, (self.x_left-5, self.y_left-5), (self.x_left+5, self.y_left+5), (0,255,255), 4)
            cv2.rectangle(line_img, (self.x_right-5, self.y_right-5), (self.x_right+5, self.y_right+5), (0,255,255), 4)
            
        elif left_x:
            self.x_left = sum(left_x) / len(left_x)
            self.x_midpoint = self.x_left + (self.prev_x_midpoint - self.prev_x_left)
            self.x_midpoint = (self.x_midpoint + self.x_left) // 2	
            cv2.rectangle(line_img, (self.x_left-5, self.y_left-5), (self.x_left+5, self.y_left+5), (0,255,255), 4)
            cv2.rectangle(line_img, (self.prev_x_right-5, self.prev_y_right-5), (self.prev_x_right+5, self.prev_y_right+5), (0,0,255), 4)
            
        elif right_x:
            self.x_right = sum(right_x) / len(right_x)
            self.x_midpoint = self.x_right + (self.prev_x_midpoint - self.prev_x_right)
            cv2.rectangle(line_img, (self.x_right-5, self.y_right-5), (self.x_right+5, self.y_right+5), (0,255,255), 4)
            cv2.rectangle(line_img, (self.prev_x_left-5, self.prev_y_left-5), (self.prev_x_left+5, self.prev_y_left+5), (0,0,255), 4)
                    
        else:
            return None

        cv2.rectangle(line_img, (self.x_midpoint-5, self.y_midpoint-5), (self.x_midpoint+5, self.y_midpoint+5), (255,0,0), 4)
        display_img[ROI_ROW:HEIGHT-ROI_OFFSET, 0:WIDTH] = line_img
        cv2.imshow('Camera', display_img)
        cv2.waitKey(1)
        
        self.prev_x_left = self.x_left
        self.prev_y_left = self.y_left
        self.prev_x_right = self.x_right
        self.prev_y_right = self.y_right
        self.prev_x_midpoint = self.x_midpoint
        self.prev_y_midpoint = self.y_midpoint
        
        return self.x_midpoint

    def detect_crosswalk(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()

        horiz_line_threshold = 10
        crosswalk_flag = None

        _, _, horiz_line_num = self.img_proc.find_line_visualize()

        if horiz_line_num > horiz_line_threshold:
            crosswalk_flag = 1
        else:
            crosswalk_flag = 0

        return crosswalk_flag


# LINE TRACKING BY USING LIDAR
# USAGE : OBSTACLE AVOIDANCE(MISSION 3), TUNNEL DRIVING(MISSION 4)
class LIDAR_DRIVING:
    def __init__(self):
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        
        self.wall_centers_pub = rospy.Publisher('/wall_centers', Marker, queue_size=10)
        self.lidar_xy_points_pub_right = rospy.Publisher('/lidar_xy_points_right', Marker, queue_size=10)
        self.lidar_xy_points_pub_left = rospy.Publisher('/lidar_xy_points_left', Marker, queue_size=10)
        self.clustered_points_pub_right = rospy.Publisher('/clustered_points_right', Marker, queue_size=10)
        self.clustered_points_pub_left = rospy.Publisher('/clustered_points_left', Marker, queue_size=10)

        self.lidar_points = None
        self.THETA2INPUT = 50 / 90 # theta : -90 ~ 90, input : -50 ~ 50
        self.LIDAR_ROI = [(0, 181), (180, 361)]
        self.lidar_ready = False
        
    def lidar_callback(self, data):
        self.lidar_points = np.asarray(data.ranges)
        self.lidar_ready = True

    def preprocess_lidar_data(self):
        if self.lidar_points is None:
            return None

        ranges = np.array(self.lidar_points)
        valid_idx = (ranges > 0.1) & (ranges < 10.0)
        points = np.column_stack((ranges[valid_idx] * np.cos(np.linspace(0, 2 * np.pi, len(ranges))[valid_idx]),
                                ranges[valid_idx] * np.sin(np.linspace(0, 2 * np.pi, len(ranges))[valid_idx])))
        return points

    def cluster(self, points):
        if len(points) < 1:
            return None, np.array([[0, 0]])

        db = DBSCAN(eps=0.5, min_samples=1).fit(points)
        labels = db.labels_

        clusters = [points[labels == label] for label in set(labels) if label != -1]
        if not clusters:
            return None, np.array([[0, 0]])

        largest_cluster = max(clusters, key=len)
        center = np.mean(largest_cluster, axis=0)
        return center, largest_cluster

    # FOR OBSTACLE AVOIDANCE(MISSION 3)
    def find_obstacle(self):
        while not self.lidar_ready:
            RATE.sleep()

        points = self.preprocess_lidar_data()
        center = self.cluster(points)

        xycacr2obstacle_vec = np.asarray(center)
        xycacr2obstacle_theta = np.degrees(np.arctan2(xycacr2obstacle_vec[0], xycacr2obstacle_vec[1]))*self.THETA2INPUT
        return xycacr2obstacle_theta
     
    # FOR TUNNEL DRIVING(MISSION 4)
    def find_midpoint(self):
        while not self.lidar_ready:
            RATE.sleep()

        points = self.preprocess_lidar_data()
        left_points = points[points[:, 0] > 0]
        right_points = points[points[:, 0] < 0]

        self.publish_lidar_points(right_points, "right")
        self.publish_lidar_points(left_points, "left")

        if len(right_points) == 0:
                right_wall_center = np.array([0, 0])
                right_cluster = np.array([[0, 0]])
        else:
            right_wall_center, right_cluster = self.cluster(right_points)
        self.publish_clustered_points(right_cluster, "right")

        if len(left_points) == 0:
            left_wall_center = np.array([0, 0])
            left_cluster = np.array([[0, 0]])
        else:
            left_wall_center, left_cluster = self.cluster(left_points)
        self.publish_clustered_points(left_cluster, "left")
            
        total_center_x = (right_wall_center[0] + left_wall_center[0])/2

        self.publish_wall_centers(right_wall_center, left_wall_center, (right_wall_center+left_wall_center)/2)

        return -total_center_x

    def publish_lidar_points(self, points, value):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        for point in points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0
            marker.points.append(p)

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        if value == "right":
            self.lidar_xy_points_pub_right.publish(marker)
        elif value == "left":
            self.lidar_xy_points_pub_left.publish(marker)

    def publish_wall_centers(self, right_wall_center, left_wall_center, total_wall_center):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # RIGHT
        right_point = Point()
        right_point.x = right_wall_center[0]
        right_point.y = right_wall_center[1]
        right_point.z = 0.1
        marker.points.append(right_point)

        # LEFT
        left_point = Point()
        left_point.x = left_wall_center[0]
        left_point.y = left_wall_center[1]
        left_point.z = 0.1
        marker.points.append(left_point)

        # MID
        mid_point = Point()
        mid_point.x = total_wall_center[0]
        mid_point.y = total_wall_center[1]
        mid_point.z = 0.1
        marker.points.append(mid_point)

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.wall_centers_pub.publish(marker)

    def publish_clustered_points(self, clustered_points, value):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        for point in clustered_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.05
            marker.points.append(p)

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        if value == "right":
            self.clustered_points_pub_right.publish(marker)
        elif value == "left":
            self.clustered_points_pub_left.publish(marker)

# MAIN LOOP
if __name__ == '__main__':
    xycar = CONTROL()
    cam_drive = CAM_DRIVING()
    lidar_drive = LIDAR_DRIVING()
    ar_tag = AR_TAG()
    traffic_light = TRAFFIC_LIGHT()
    
    while not rospy.is_shutdown():
        ar_ID, ar_distance = ar_tag.AR_detect()
        # cam_midpoint = cam_drive.find_midpoint_visualize()
        
        # if cam_midpoint is None:
        #     midpoint = lidar_drive.find_midpoint()
        # else:
        #     midpoint = cam_midpoint
        
        # midpoint = cam_midpoint
        midpoint = lidar_drive.find_midpoint()
        if midpoint is not None:
            # angle = xycar.pid(midpoint, P_GAIN, I_GAIN, D_GAIN, "LINE_TRACKING")
            angle = xycar.pid(midpoint, "TUNNEL DRIVING")
            speed = 0 # Adjust speed as necessary
            xycar.drive(angle, speed)
        
        # RATE.sleep()

        if ar_ID == 2:
            gostop, direction = traffic_light.traffic_crossroad()
            if ar_distance < 0.1:
                speed = 0

        if ar_ID == 4:
            gostop = traffic_light.traffic_single()

        if ar_ID == 6 and ar_distance < 0.1:
            midpoint = lidar_drive.find_midpoint()