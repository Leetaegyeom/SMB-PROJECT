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
ROI_ROW = 360
ROI_OFFSET = 90

P_GAIN_CAM = 0.6
I_GAIN_CAM = 0.006
D_GAIN_CAM = 0.001

P_GAIN_TUNNEL = 7.0
I_GAIN_TUNNEL = 0.0
D_GAIN_TUNNEL = 7.0

P_GAIN_OBS = 1.0
I_GAIN_OBS = 0.0
D_GAIN_OBS = 0.0

SPEED = 5
DELTA_50 = 50
TUNNEL_OUT_THRESHOLD = 10

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
        
    def pid(self, input_data, type="LINE TRACKING"):
        # print(input_data)
        if type == "LINE TRACKING":
            error = WIDTH // 2 - input_data
            self.kp = P_GAIN_CAM
            self.ki = I_GAIN_CAM
            self.kd = D_GAIN_CAM

        elif type == "AVOID OBSTACLE":
            error = DELTA_50 - input_data
            self.kp = P_GAIN_OBS
            self.ki = I_GAIN_OBS
            self.kd = D_GAIN_OBS

        elif type == "TUNNEL DRIVING":
            error = DELTA_50 - input_data
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
        
        # CLAHE 적용
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # hist_equalized = clahe.apply(gray)
        
        # HE 적용
        hist_equalized = cv2.equalizeHist(gray)
        
        # GaussianBlur
        blur_gray = cv2.GaussianBlur(hist_equalized, (5, 5), 0)
        
        # Canny Edge Detection
        edge_img = cv2.Canny(np.uint8(blur_gray), 150, 500)
        roi_edge_img = edge_img[ROI_ROW:HEIGHT, 0:WIDTH]

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
                
            elif slope > 0.2 and x1 > WIDTH / 2:
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
        # min_distance = float('inf')
        min_distance = 0.5

        if len(self.arData["ID"]) == 0:
            return None, min_distance
        
        for idx, distance in enumerate(self.arData["DZ"]):
            if distance < min_distance:
                min_distance = distance
                min_ID = self.arData["ID"][idx]
                
        return min_ID, min_distance


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
            print("There is no Signal from Single Color!!!")
            return 0

        if self.single_color == 'G':
            gostop = 1
        
        elif self.single_color == 'Y':
            if self.prev_single_color == 'G':
                gostop = 1
            
            elif self.prev_single_color == 'Y':
                gostop = 0
            
        elif self.single_color == 'R':
            gostop = 0

        return gostop
    
    def traffic_crossroad(self):
        direction = None
        gostop = None

        if self.left_color is None or self.right_color is None or self.time_count is None:
            return None, None

        if self.time_count >= 3:
            gostop = 1
        else:
            gostop = 0

        if self.left_color == 'R':
            direction = 30
        else:
            direction = -30

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
            self.x_midpoint = 0

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
            cv2.rectangle(line_img, (self.x_left-5, self.y_left-5), (self.x_left+5, self.y_left+5), (0,255,255), 4)
            cv2.rectangle(line_img, (self.prev_x_right-5, self.prev_y_right-5), (self.prev_x_right+5, self.prev_y_right+5), (0,0,255), 4)

        elif right_x:
            self.x_right = sum(right_x) / len(right_x)
            self.x_midpoint = self.x_right + (self.prev_x_midpoint - self.prev_x_right)
            cv2.rectangle(line_img, (self.x_right-5, self.y_right-5), (self.x_right+5, self.y_right+5), (0,255,255), 4)
            cv2.rectangle(line_img, (self.prev_x_left-5, self.prev_y_left-5), (self.prev_x_left+5, self.prev_y_left+5), (0,0,255), 4)

        else:
            self.x_midpoint = 0
            cv2.rectangle(line_img, (self.prev_x_right-5, self.prev_y_right-5), (self.prev_x_right+5, self.prev_y_right+5), (0,0,255), 4)
            cv2.rectangle(line_img, (self.prev_x_right-5, self.prev_y_right-5), (self.prev_x_right+5, self.prev_y_right+5), (0,0,255), 4)
            
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

        _, _, _, _, horiz_line_num = self.img_proc.find_line()

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
        
        # self.wall_centers_pub = rospy.Publisher('/wall_centers', Marker, queue_size=10)
        self.lidar_xy_points_pub = rospy.Publisher('/lidar_xy_points', Marker, queue_size=10)
        self.clustered_points_pub = rospy.Publisher('/clustered_points', Marker, queue_size=10)
        self.vector_pub = rospy.Publisher('/min_max_vector', Marker, queue_size=10)
        self.min_max_point_pub = rospy.Publisher('/min_max_point', Marker, queue_size=10)

        self.lidar_points = None
        self.THETA2INPUT = 50.0 / 90.0 # theta : -90 ~ 90, input : -50 ~ 50
        self.LIDAR_ROI = [(0, 181), (180, 361)]
        self.lidar_ready = False
        
    def lidar_callback(self, data):
        self.lidar_points = np.asarray(data.ranges)
        self.lidar_ready = True

    def preprocess_lidar_data(self):
        if self.lidar_points is None:
            return None

        ranges = np.array(self.lidar_points)
        valid_idx = (ranges > 0.1) & (ranges < 0.70)
        points = np.column_stack((ranges[valid_idx] * np.cos(np.linspace(0, 2 * np.pi, len(ranges))[valid_idx]),
                                ranges[valid_idx] * np.sin(np.linspace(0, 2 * np.pi, len(ranges))[valid_idx])))
        return points

    def cluster(self, points):
        if len(points) < 1:
            return None, np.array([[0, 0]]), None, None, None

        db = DBSCAN(eps=0.5, min_samples=1).fit(points)
        labels = db.labels_

        clusters = [points[labels == label] for label in set(labels) if label != -1]
        if not clusters:
            return None, np.array([[0, 0]]), None, None, None

        largest_cluster = max(clusters, key=len)
        center = np.mean(largest_cluster, axis=0)

        largest_cluster_indices = np.where(labels == np.argmax(np.bincount(labels[labels != -1])))[0]

        max_y_index = largest_cluster_indices[np.argmax(largest_cluster[:, 1])]
        min_y_index = largest_cluster_indices[np.argmin(largest_cluster[:, 1])]

        cluster_centers = [np.mean(cluster, axis=0) for cluster in clusters]

        origin = np.array([0, 0])
        distances_to_origin = [np.linalg.norm(center - origin) for center in cluster_centers]
        closest_cluster_index = np.argmin(distances_to_origin)
        closest_cluster = clusters[closest_cluster_index] if closest_cluster_index < len(clusters) else None

        return center, largest_cluster, max_y_index, min_y_index, closest_cluster

    # FOR OBSTACLE AVOIDANCE(MISSION 3)
    def find_obstacle(self):
        while not self.lidar_ready:
            RATE.sleep()

        points = self.preprocess_lidar_data()
        obs_center, obs_clusters, _, _, _ = self.cluster(points)

        xycacr2obs_vec = np.asarray(obs_center)
        xycacr2obs_theta = np.degrees(np.arctan2(xycacr2obs_vec[1], xycacr2obs_vec[0])) * self.THETA2INPUT
        return xycacr2obs_theta

    def find_midpoint(self):
        while not self.lidar_ready:
            RATE.sleep()

        points = self.preprocess_lidar_data()
        self.publish_lidar_points(points)
        _, clusters, ymax_idx, ymin_idx, _ = self.cluster(points)
        self.publish_clustered_points(clusters)
        max_point = points[ymax_idx]
        min_point = points[ymin_idx]
        print(ymax_idx)
        
        self.publish_min_max_point(min_point, max_point)

        min_max_vec = max_point - min_point
        self.publish_ref_vector(min_max_vec)

        ref_angle = np.degrees(np.arctan2(min_max_vec[1], min_max_vec[0])) * self.THETA2INPUT
        return ref_angle

    def find_closest_cluster(self):
        while not self.lidar_ready:
            RATE.sleep()
        
        points = self.preprocess_lidar_data()
        _, _, _, _, closest_cluster = self.cluster(points)
        closest_cluster_center = np.mean(closest_cluster, axis=0) if closest_cluster is not None else None
        
        if closest_cluster_center is not None:
            distance = np.linalg.norm(closest_cluster_center - np.array([0, 0]))
        else:
            distance = float("inf")
        
        return closest_cluster_center, distance

    def publish_lidar_points(self, points):
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
        self.lidar_xy_points_pub.publish(marker)

    # def publish_wall_centers(self, right_wall_center, left_wall_center):
    #     marker = Marker()
    #     marker.header.frame_id = "base_link"
    #     marker.type = Marker.POINTS
    #     marker.action = Marker.ADD

    #     # RIGHT
    #     right_point = Point()
    #     right_point.x = right_wall_center[0]
    #     right_point.y = right_wall_center[1]
    #     right_point.z = 0.1
    #     marker.points.append(right_point)

    #     # LEFT
    #     left_point = Point()
    #     left_point.x = left_wall_center[0]
    #     left_point.y = left_wall_center[1]
    #     left_point.z = 0.1
    #     marker.points.append(left_point)

    #     marker.scale.x = 0.2
    #     marker.scale.y = 0.2
    #     marker.scale.z = 0.2
    #     marker.color.a = 1.0
    #     marker.color.r = 1.0
    #     marker.color.g = 0.0
    #     marker.color.b = 0.0

    #     self.wall_centers_pub.publish(marker)

    def publish_clustered_points(self, clustered_points):
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
        self.clustered_points_pub.publish(marker)

    def publish_ref_vector(self, vector):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Start point of the vector
        start_point = Point()
        start_point.x = 0
        start_point.y = 0
        start_point.z = 0

        # End point of the vector
        end_point = Point()
        end_point.x = vector[0]
        end_point.y = -vector[1]
        end_point.z = 0  # Adjust if needed

        marker.points.append(start_point)
        marker.points.append(end_point)

        marker.scale.x = 0.1
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0
        marker.color.r = 1.0 
        marker.color.g = 0.0 
        marker.color.b = 0.0
        self.vector_pub.publish(marker)

    def publish_min_max_point(self, min_point_, max_point_):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        # print(min_point)
        # MIN
        min_point = Point()
        min_point.x = min_point_[0]
        min_point.y = min_point_[1]
        min_point.z = 0.2
        marker.points.append(min_point)

        # MAX
        max_point = Point()
        max_point.x = max_point_[0]
        max_point.y = max_point_[1]
        max_point.z = 0.1
        marker.points.append(max_point)

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.min_max_point_pub.publish(marker)

# MAIN LOOP
if __name__ == '__main__':
    xycar = CONTROL()
    cam_drive = CAM_DRIVING()
    lidar_drive = LIDAR_DRIVING()
    ar_tag = AR_TAG()
    traffic_light = TRAFFIC_LIGHT()

    speed = 0
    drive_mode = "CAM"      # Cam or Lidar mode
    
    while not rospy.is_shutdown():
        # ar_ID, ar_distance = ar_tag.AR_detect()
        # crosswalk_flag = cam_drive.detect_crosswalk()
        # cam_midpoint = cam_drive.find_midpoint_visualize()
        
        # if cam_midpoint is None:
        #     midpoint = lidar_drive.find_midpoint()
        # else:
        #     midpoint = cam_midpoint
        
        # midpoint = cam_midpoint
        
        ########## Lidar 클래스에서 젤 가까운 클러스터 위치, 거리 받아오는 알고리즘 ##########
        closest_cluster_center, distance = lidar_drive.find_closest_cluster()
        ##################################################################################

        #####################[CAM DRIVE TEST]#############################
        # midpoint = cam_drive.find_midpoint_visualize()
        # if midpoint is not None:
        #     angle = xycar.pid(midpoint, "LINE TRACKING")
        #     speed = 5
        #     xycar.drive(angle, speed)
        #########################################################

        ######################[TUNNEL DRIVE TEST]###################################
        midpoint = lidar_drive.find_midpoint()
        if midpoint is not None:
            angle = xycar.pid(midpoint, "TUNNEL DRIVING") -20
            speed = 5
            xycar.drive(angle, speed)
        ############################################################################

        ######################[AVOID OBSTACLE TEST]###################################
        # obs_xycar_theta = lidar_drive.find_obstacle()
        # angle = xycar.pid(obs_xycar_theta, "AVOID OBSTACLE")
        # speed = 5 # Adjust speed as necessary
        # xycar.drive(angle, speed)
        ###########################################################################

        # if ar_ID is not None and ar_distance is not None:
        #     if ar_ID == 4:
        #         gostop, direction = traffic_light.traffic_crossroad()
        #         if ar_distance < 0.1:
        #             speed = 0
        #         elif direction is not None:
        #             if direction == 'left':

        #                 pass
                    
        #             elif direction == 'right':

        #                 pass
        #     elif ar_ID == 2:
        #         gostop = traffic_light.traffic_single()

        # if crosswalk_flag == 1:
        #     if gostop == 'stop':
        #         speed = 0

        # xycar.drive()
        RATE.sleep()

# MAIN LOOP REAL Ver.
if __name__ == '__main__':
    xycar = CONTROL()
    cam_drive = CAM_DRIVING()
    lidar_drive = LIDAR_DRIVING()
    ar_tag = AR_TAG()
    traffic_light = TRAFFIC_LIGHT()

    speed = 5
    can_we_go = 1
    is_done = False
    is_single_color = True
    drive_mode = "CAM"      # Camera or Lidar mode
    
    # Need to delete
    prev_mode = drive_mode
    prev_ar_ID = 0
    prev_crw_flag = 0
    prev_cluster_distance = float("inf")
                
    while not rospy.is_shutdown():
        # AR Detect
        ar_ID, ar_distance = ar_tag.AR_detect()
        
        # Detect Crosswalk
        crosswalk_flag = cam_drive.detect_crosswalk()
        
        # Find Closest Cluster
        closest_cluster_center, cluster_distance = lidar_drive.find_closest_cluster()
        
        if ar_ID and ar_ID == 6 and drive_mode == "CAM":
            pass_stack = 0
            drive_mode = "LIDAR"

        # Find Midpoint to follow
        if drive_mode == "CAM":
            midpoint = cam_drive.find_midpoint_visualize()
            
            # Avoid Obstacle
            if ar_ID is None and cluster_distance < 0.3:
                midpoint += closest_cluster_center[0]
                
        # Tunnel Mission
        else:
            midpoint = lidar_drive.find_midpoint()
            
            if closest_cluster_center is None:
                pass_stack += 1
            
            # Finish Tunnel
            if pass_stack > TUNNEL_OUT_THRESHOLD:
                drive_mode = "CAM"
                
        # Crosswalk
        if crosswalk_flag and is_single_color:
            can_we_go = traffic_light.traffic_single()
            
            if can_we_go == 1:
                is_single_color = False
        
        # Crossroads & Stop Mission
        if ar_ID:
            if ar_ID == 2:
                can_we_go, direction = traffic_light.traffic_crossroad()
                
                if can_we_go == 1:
                    for _ in range(5):
                        xycar.drive(0, speed)
                        RATE.sleep()
                    for _ in range(10):
                        xycar.drive(direction, speed)
                        RATE.sleep()
                    
                    continue    # Find Line Again
            
            elif ar_ID == 4:
                while True:
                    _, ar_distance = ar_tag.AR_detect()
                    
                    # Need to make Angle Calculate Module !!!
                    # 
                    # 
                    # # # # # # # # # # # # # # # # # #
                
                    if ar_distance < 0.2:
                        is_done = True
                        break
                    
                    xycar.drive(0, speed)
                    RATE.sleep()
        
        if is_done:
            print("Race is done.")
            break
            
        angle = xycar.pid(midpoint)
        xycar.drive(angle, speed * can_we_go)
        
        if prev_mode != drive_mode or prev_ar_ID != ar_ID or prev_crw_flag != crosswalk_flag or prev_cluster_distance != cluster_distance:
            print(f"Present Mode: {drive_mode}")
            print(f"Present AR ID: {ar_ID}")
            print(f"Present Crosswalk Flag: {crosswalk_flag}")            
            if cluster_distance == float("inf"):
                print("There is no Cluster!!!")
            else:
                print(f"Present Cluster Distance: {cluster_distance:.2f}")
        
        prev_mode = drive_mode
        prev_ar_ID = ar_ID
        prev_crw_flag = crosswalk_flag
        prev_cluster_distance = cluster_distance
        
        # RATE.sleep()