#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary libraries
import numpy as np
import cv2, math
import rospy
from cv_bridge import CvBridge
from sklearn.cluster import DBSCAN

# Import message types from ROS
from std_msgs.msg import Int64, String
from sensor_msgs.msg import Image, LaserScan
from xycar_motor.msg import xycar_motor
from ar_track_alvar_msgs.msg import AlvarMarkers
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3

# Initialize the ROS node
rospy.init_node('xycar')

# Control parameters and constants
CONTROL_TIME = 0.1
RATE = rospy.Rate(1 / CONTROL_TIME)
WIDTH, HEIGHT = 640, 480
ROI_ROW = 240
ROI_OFFSET = 50

# PID control gains for camera and tunnel driving
P_GAIN_CAM = 0.6
I_GAIN_CAM = 0.006
D_GAIN_CAM = 0.001

P_GAIN_TUNNEL = 7.0
I_GAIN_TUNNEL = 0.0
D_GAIN_TUNNEL = 7.0

# Delta offset for tunnel driving
DELTA_50 = 50   # Set to 55 for left offset
TUNNEL_OUT_THRESHOLD = 10


# Class to control the Xycar
class CONTROL:
    def __init__(self):
        # Initialize the publisher for motor control
        self.motor_pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
        
        # Initialize PID errors
        self.i_error = 0.0
        self.prev_error = 0.0
        self.ANGLE_LIMIT = 50
        
        # Initialize PID gains
        self.kp = 0
        self.ki = 0
        self.kd = 0
    
    # PID control function
    def pid(self, input_data, type="LINE TRACKING"):
        if type == "LINE TRACKING":
            error = WIDTH // 2 - input_data
            self.kp = P_GAIN_CAM
            self.ki = I_GAIN_CAM
            self.kd = D_GAIN_CAM

        elif type == "TUNNEL DRIVING":
            error = DELTA_50 - input_data
            self.kp = P_GAIN_TUNNEL
            self.ki = I_GAIN_TUNNEL
            self.kd = D_GAIN_TUNNEL
            
        # Calculate derivative error
        derror = error - self.prev_error

        # Proportional, integral, and derivative calculations
        p_error = self.kp * error
        self.i_error = self.i_error + self.ki * error * CONTROL_TIME
        d_error = self.kd * derror / CONTROL_TIME

        # Compute the output control signal
        output = p_error + self.i_error + d_error
        self.prev_error = error

        # Limit the output to the defined angle limits
        if output > self.ANGLE_LIMIT:
            output = self.ANGLE_LIMIT
        elif output < -self.ANGLE_LIMIT:
            output = -self.ANGLE_LIMIT
        return -output

    # Function to send drive commands to Xycar
    def drive(self, Angle, Speed):
        motor_msg = xycar_motor()
        motor_msg.angle = Angle
        motor_msg.speed = Speed
        self.motor_pub.publish(motor_msg)


# IMAGE PROCESSING FOR CAM_DRIVING
class IMG_PROCESSING:
    def __init__(self):
        # Subscribe to the camera image topic
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        
        # Initialize image data and CV bridge
        self.image = np.empty(shape=[0])
        self.bridge = CvBridge()
        self.img_ready = False
        
    # Callback function to process image data
    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.img_ready = True
        
    # Check if the image is ready
    def is_image_ready(self):
        return self.img_ready and self.image.size == (WIDTH * HEIGHT * 3)
        
    # Find the lane line in the image
    def find_line(self):        
        img = self.image.copy()
        self.img_ready = False

        # Convert to grayscale and equalize histogram
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist_equalized = cv2.equalizeHist(gray)
        
        # GaussianBlur
        blur_gray = cv2.GaussianBlur(hist_equalized, (5, 5), 0)
        
        # Canny Edge Detection
        edge_img = cv2.Canny(np.uint8(blur_gray), 150, 500)
        roi_edge_img = edge_img[ROI_ROW:HEIGHT-ROI_OFFSET, 0:WIDTH]
        
        # For Visualize
        display_img = img
        line_img = img.copy()[ROI_ROW:HEIGHT-ROI_OFFSET, 0:WIDTH]
        cv2.rectangle(line_img, (5, ROI_ROW+5), (WIDTH-5, HEIGHT-ROI_OFFSET-5), (0,255,0), 2)

        # Detect lines using Hough Transform
        all_lines = cv2.HoughLinesP(roi_edge_img, 1, math.pi/180, 50, 50, 20)
        if all_lines is None:
            return [], [], [], [], 0

        left_x, right_x = [], []
        left_y, right_y = [], []

        # Separate left and right lane lines based on slope
        for line in all_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < -0.2 and x2 < WIDTH / 2:
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            elif slope > 0.2 and x1 > WIDTH / 2:
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Visualize the result
        display_img[ROI_ROW:HEIGHT-ROI_OFFSET, 0:WIDTH] = line_img
        cv2.line(display_img, (0, (HEIGHT - ROI_OFFSET + ROI_ROW)/2), (WIDTH, (HEIGHT - ROI_OFFSET + ROI_ROW)/2), (0, 255, 255), 2)
        cv2.imshow('camera', display_img)
        cv2.waitKey(1)

        # For Detect Crosswalk
        roi_gray = gray[380:HEIGHT, 60:WIDTH-60]
        _, binary_img = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY)
        
        # Calculate the white pixel ratio
        white_pixel_count = cv2.countNonZero(binary_img)
        total_pixels = binary_img.shape[0] * binary_img.shape[1]
        white_pixel_ratio = float(white_pixel_count) / float(total_pixels)

        # Determine if it's a crosswalk based on the white pixel ratio
        is_crosswalk = white_pixel_ratio > 0.4

        return left_x, right_x, left_y, right_y, is_crosswalk

    # Return the current image
    def get_img(self):
        return self.image.copy()
    
    
# AR TAG DETECTION & IDENTIFICATION
class AR_TAG:
    def __init__(self):
        # Initialize AR tag data
        self.arData = {"ID":[], "DZ":[]}

        # Subscribe to AR tag pose marker topic
        rospy.Subscriber('/ar_pose_marker', AlvarMarkers, self.AR_callback)

    # Callback function to process AR tag data
    def AR_callback(self, data):
        self.arData = {"ID":[], "DZ":[]}

        for i in data.markers:
            self.arData["ID"].append(i.id)
            self.arData["DZ"].append(i.pose.pose.position.z)

    # Detect the closest AR tag
    def AR_detect(self):
        min_ID = None
        min_distance = float('inf')

        if len(self.arData["ID"]) == 0:
            return min_ID, float("inf")
        
        for idx, distance in enumerate(self.arData["DZ"]):
            if distance < min_distance:
                min_distance = distance
                min_ID = self.arData["ID"][idx]
                
        return min_ID, min_distance

# TRAFFIC LIGHT 
class TRAFFIC_LIGHT:
    def __init__(self):
        # Initialize traffic light color data
        self.single_color = None
        self.prev_single_color = None
        self.right_color = None
        self.left_color = None
        self.time_count = None

        # Subscribe to traffic light color and time topics
        rospy.Subscriber("/Single_color", String, self.single_callback)
        rospy.Subscriber("/Right_color", String, self.right_callback)
        rospy.Subscriber("/Left_color", String, self.left_callback)
        rospy.Subscriber("/time_count", Int64, self.time_callback)

    # Callback function for single color traffic light
    def single_callback(self, data):
        self.prev_single_color = self.single_color
        self.single_color = data.data

    # Callback function for right turn traffic light
    def right_callback(self, data):
        self.right_color = data.data

    # Callback function for left turn traffic light
    def left_callback(self, data):
        self.left_color = data.data

    # Callback function for time count
    def time_callback(self, data):
        self.time_count = data.data

    # Determine if the vehicle should go or stop based on single color traffic light
    def traffic_single(self):
        gostop = 0

        if self.single_color is None:
            print("There is no Signal from Single Color!!!")
            return gostop

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
    
    # Determine the direction and if the vehicle should go or stop at a crossroad
    def traffic_crossroad(self):
        direction = None
        gostop = 1

        if self.left_color is None or self.right_color is None or self.time_count is None:
            print("There is no Traffic Light!!!")
            return gostop, direction

        if self.left_color == 'G' or self.left_color == 'Y':
            if self.time_count >= 7:
                direction = "left"
            else:
                direction = "right"
                
        if self.right_color == 'G' or self.right_color == 'Y':
            if self.time_count >= 7:
                direction = "right"
            else:
                direction = "left"

        return gostop, direction
        

# LINE TRACKING BY USING CAMERA
# USAGE: TWO LINE TRACKING(MISSION 1), ONE LINE TRACKING(MISSION 2)
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

        self.left_left_flag = False
        self.left_right_flag = False
        self.right_left_flag = False
        self.right_right_flag = False
        self.total_flag = False

    # Find the midpoint of the lane lines
    def find_midpoint(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()

        left_x, right_x, _, _, is_crosswalk = self.img_proc.find_line()

        if left_x and right_x:
            self.x_left = sum(left_x) / len(left_x)
            self.x_right = sum(right_x) / len(right_x)
            self.x_midpoint = (self.x_left + self.x_right) / 2

        elif left_x:
            self.x_left = sum(left_x) / len(left_x)
            self.x_midpoint = self.x_left + (self.prev_x_midpoint - self.prev_x_left)

        elif right_x:
            self.x_right = sum(right_x) / len(right_x)
            self.x_midpoint = self.x_right + (self.prev_x_midpoint - self.prev_x_right)

        else:
            self.x_midpoint = self.prev_x_midpoint

        self.prev_x_left = self.x_left
        self.prev_x_right = self.x_right
        self.prev_x_midpoint = self.x_midpoint

        return self.x_midpoint, is_crosswalk

    # Check if the car is at a crossroad
    def check_crossroad(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()
        
        left_x, right_x, left_y, right_y, _ = self.img_proc.find_line()

        for x1, x2, y1, y2 in zip(left_x[::2], left_x[1::2], left_y[::2], left_y[1::2]):
            y1 += ROI_ROW
            y2 += ROI_ROW
            if x1 < float(WIDTH) / 4.0 and x2 < float(WIDTH) / 4.0:
                if y1 < float(ROI_ROW + HEIGHT - ROI_OFFSET) / 2.0 or y2 < float(ROI_ROW + HEIGHT - ROI_OFFSET) / 2.0:
                    self.left_left_flag = True
            elif x1 > float(WIDTH) / 4.0 and x2 > float(WIDTH) / 4.0:
                if y1 < float(ROI_ROW + HEIGHT - ROI_OFFSET) / 2.0 or y2 < float(ROI_ROW + HEIGHT - ROI_OFFSET) / 2.0:
                    self.left_right_flag = True
        
        for x1, x2, y1, y2 in zip(right_x[::2], right_x[1::2], right_y[::2], right_y[1::2]):
            if x1 < float(WIDTH) * (3.0/4.0) and x2 < float(WIDTH) * (3.0/4.0):
                if y1 < float(ROI_ROW + HEIGHT - ROI_OFFSET) / 2.0 or y2 < float(ROI_ROW + HEIGHT - ROI_OFFSET) / 2.0:
                    self.right_left_flag = True
            elif x1 > float(WIDTH) * (3.0/4.0) and x2 > float(WIDTH) * (3.0/4.0):
                if y1 < float(ROI_ROW + HEIGHT - ROI_OFFSET) / 2.0 or y2 < float(ROI_ROW + HEIGHT - ROI_OFFSET) / 2.0:
                    self.right_right_flag = True
            
        if self.left_left_flag and self.left_right_flag and self.right_left_flag and self.right_right_flag:
            self.total_flag = True
            
        self.left_left_flag = self.left_right_flag = self.right_left_flag = self.right_right_flag = False
        
        return self.total_flag

    # Find midpoint for crosswalk (left)
    def find_midpoint_crosswalk_left(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()

        left_of_left = []
        right_of_left = []
        left_of_right = []
        right_of_right = []
        direction_flag = 0
        left_x, right_x, _, _, _ = self.img_proc.find_line()

        for i in range(0, len(left_x), 2):
            if left_x[i] < float(WIDTH) / 4.0 and left_x[i+1] < float(WIDTH) / 4.0:
                left_of_left.append(left_x[i])
                left_of_left.append(left_x[i+1])
            elif left_x[i] > float(WIDTH) / 4.0 and left_x[i+1] > float(WIDTH) / 4.0:
                right_of_left.append(left_x[i])
                right_of_left.append(left_x[i+1])

        for i in range(0, len(right_x), 2):
            if right_x[i] < float(WIDTH) * (3.0 / 4.0) and right_x[i+1] < float(WIDTH) * (3.0 / 4.0):
                left_of_right.append(right_x[i])
                left_of_right.append(right_x[i+1])
            elif right_x[i] > float(WIDTH) * (3.0 / 4.0) and right_x[i+1] > float(WIDTH) * (3.0 / 4.0):
                right_of_right.append(right_x[i])
                right_of_right.append(right_x[i+1])

        if left_of_left and right_of_left:
            self.x_left = sum(left_of_left) / len(left_of_left)
            self.x_right = sum(right_of_left) / len(right_of_left)
            self.x_midpoint = (self.x_left + self.x_right) // 2 - 20

        elif left_of_left:
            self.x_left = sum(left_of_left) / len(left_of_left)
            self.x_midpoint = self.x_left + (self.prev_x_midpoint - self.prev_x_left)

        else:
            self.x_midpoint = self.prev_x_midpoint       

        self.prev_x_left = self.x_left
        self.prev_x_right = self.x_right
        self.prev_x_midpoint = self.x_midpoint

        return self.x_midpoint, direction_flag
    
    # Find midpoint for crosswalk (right)
    def find_midpoint_crosswalk_right(self):
        while not self.img_proc.is_image_ready():
            RATE.sleep()

        left_of_left = []
        right_of_left = []
        left_of_right = []
        right_of_right = []
        direction_flag = 0
        left_x, right_x, _, _, _ = self.img_proc.find_line()

        for i in range(0, len(left_x), 2):
            if left_x[i] < float(WIDTH) / 4.0 and left_x[i+1] < float(WIDTH) / 4.0:
                left_of_left.append(left_x[i])
                left_of_left.append(left_x[i+1])
            elif left_x[i] > float(WIDTH) / 4.0 and left_x[i+1] > float(WIDTH) / 4.0:
                right_of_left.append(left_x[i])
                right_of_left.append(left_x[i+1])

        for i in range(0, len(right_x), 2):
            if right_x[i] < float(WIDTH) * (3.0 / 4.0) and right_x[i+1] < float(WIDTH) * (3.0 / 4.0):
                left_of_right.append(right_x[i])
                left_of_right.append(right_x[i+1])
            elif right_x[i] > float(WIDTH) * (3.0 / 4.0) and right_x[i+1] > float(WIDTH) * (3.0 / 4.0):
                right_of_right.append(right_x[i])
                right_of_right.append(right_x[i+1])

        if left_of_left and right_of_right:
            self.x_left = sum(left_of_left) / len(left_of_left)
            self.x_right = sum(right_of_right) / len(right_of_right)
            self.x_midpoint = (self.x_left + self.x_right) // 2 + 20

        elif right_of_right:
            self.x_right = sum(right_of_right) / len(right_of_right)
            self.x_midpoint = self.x_right + (self.prev_x_midpoint - self.prev_x_right)        
        
        else:
            self.x_midpoint = self.prev_x_midpoint          

        self.prev_x_left = self.x_left
        self.prev_x_right = self.x_right
        self.prev_x_midpoint = self.x_midpoint

        return self.x_midpoint, direction_flag


# LINE TRACKING BY USING LIDAR
# USAGE: OBSTACLE AVOIDANCE (MISSION 3), TUNNEL DRIVING (MISSION 4)
class LIDAR_DRIVING:
    def __init__(self):
        # Subscribe to the LIDAR scan topic
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        
        # Publishers for LIDAR-related markers and points
        self.lidar_xy_points_pub = rospy.Publisher('/lidar_xy_points', Marker, queue_size=10)
        self.clustered_points_pub = rospy.Publisher('/clustered_points', Marker, queue_size=10)
        self.vector_pub = rospy.Publisher('/min_max_vector', Marker, queue_size=10)
        self.min_max_point_pub = rospy.Publisher('/min_max_point', Marker, queue_size=10)

        # Initialize LIDAR points and constants
        self.lidar_points = None
        self.THETA2INPUT = 50.0 / 90.0  # Convert theta (-90 ~ 90) to input (-50 ~ 50)
        self.LIDAR_ROI = [(0, 181), (180, 361)]
        self.lidar_ready = False
        
    # Callback function for LIDAR data
    def lidar_callback(self, data):
        self.lidar_points = np.asarray(data.ranges)
        self.lidar_ready = True

    # Get side distances from LIDAR points
    def get_side_distance(self):
        if self.lidar_points is None:
            return 0, 0
        return self.lidar_points[0], self.lidar_points[360]

    # Preprocess LIDAR data for clustering
    def preprocess_lidar_data(self):
        if self.lidar_points is None:
            return np.asarray([[0, 0]])

        ranges = np.array(self.lidar_points)
        valid_idx = (ranges > 0.1) & (ranges < 0.50)
        points = np.column_stack((ranges[valid_idx] * np.cos(np.linspace(0, 2 * np.pi, len(ranges))[valid_idx]),
                                  ranges[valid_idx] * np.sin(np.linspace(0, 2 * np.pi, len(ranges))[valid_idx])))
        return points

    # Cluster LIDAR points using DBSCAN
    def cluster(self, points):
        if len(points) <= 1:
            return None, np.array([[0, 0]]), 0, 0, None

        db = DBSCAN(eps=0.5, min_samples=1).fit(points)
        labels = db.labels_

        clusters = [points[labels == label] for label in set(labels) if label != -1]
        if not clusters:
            return None, np.array([[0, 0]]), 0, 0, None

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

    # Find midpoint for tunnel driving (MISSION 4)
    def find_midpoint(self):
        while not self.lidar_ready:
            RATE.sleep()

        points = self.preprocess_lidar_data()
        if len(points) <= 1:
            return 50

        self.publish_lidar_points(points)
        center, clusters, ymax_idx, ymin_idx, _ = self.cluster(points)
        if clusters is None:
            return 50

        max_point = points[ymax_idx]
        min_point = points[ymin_idx]
        
        self.publish_min_max_point(min_point, max_point)

        min_max_vec = max_point - min_point
        self.publish_ref_vector(min_max_vec)

        ref_angle = np.degrees(np.arctan2(min_max_vec[1], min_max_vec[0])) * self.THETA2INPUT

        return ref_angle

    # Find closest cluster for obstacle avoidance (MISSION 3)
    def find_closest_cluster(self):
        while not self.lidar_ready:
            RATE.sleep()
        
        points = self.preprocess_lidar_data()
        if len(points) <= 1:
            return None, float("inf")

        _, _, _, _, closest_cluster = self.cluster(points)
        closest_cluster_center = np.mean(closest_cluster, axis=0) if closest_cluster is not None else None
        
        if closest_cluster_center is not None:
            distance = np.linalg.norm(closest_cluster_center)
            distance = round(distance, 3)
        else:
            distance = float("inf")
        
        return closest_cluster_center, distance


    # Publish LIDAR points for visualization
    def publish_lidar_points(self, points):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Add points to the marker
        for point in points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0
            marker.points.append(p)

        # Set marker properties
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        self.lidar_xy_points_pub.publish(marker)

    # Publish clustered LIDAR points for visualization
    def publish_clustered_points(self, clustered_points):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Add clustered points to the marker
        for point in clustered_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.05
            marker.points.append(p)

        # Set marker properties
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        self.clustered_points_pub.publish(marker)

    # Publish reference vector for visualization
    def publish_ref_vector(self, vector):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Define the start and end points of the vector
        start_point = Point()
        start_point.x = 0
        start_point.y = 0
        start_point.z = 0

        end_point = Point()
        end_point.x = vector[0]
        end_point.y = -vector[1]
        end_point.z = 0

        marker.points.append(start_point)
        marker.points.append(end_point)

        # Set marker properties
        marker.scale.x = 0.1
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.vector_pub.publish(marker)

    # Publish minimum and maximum points for visualization
    def publish_min_max_point(self, min_point_, max_point_):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Define the minimum point
        min_point = Point()
        min_point.x = min_point_[0]
        min_point.y = min_point_[1]
        min_point.z = 0.2
        marker.points.append(min_point)

        # Define the maximum point
        max_point = Point()
        max_point.x = max_point_[0]
        max_point.y = max_point_[1]
        max_point.z = 0.1
        marker.points.append(max_point)

        # Set marker properties
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

    angle = 0
    speed = 5
    can_we_go = 1
    stack = 0
    right_stack = 0
    start_time_2 = None
    
    is_done = False
    crosswalk_flag = False
    is_single_color = True
    tunnel_detect_flag = False
    drive_mode = "CAM"  # Camera or Lidar mode
    is_crossroad = False

    while not rospy.is_shutdown():
       
        # AR Detect
        ar_ID, ar_distance = ar_tag.AR_detect()
        
        # Find Closest Cluster
        closest_cluster_center, cluster_distance = lidar_drive.find_closest_cluster()
        drive_type = "LINE TRACKING"

        # Detect tunnel
        if ar_ID and ar_ID == 6 and ar_distance < 0.5 and drive_mode == "CAM":
            if not tunnel_detect_flag:
                print("TUNNEL DETECT!!!")
                tunnel_detect_flag = True

        # Switch to LIDAR driving mode in tunnel
        if tunnel_detect_flag:
            _, right = lidar_drive.get_side_distance()
            if right < 0.6:
                right_stack += 1
            else:
                right_stack = 0
            if cluster_distance < 0.3 and right_stack > 10:
                print("TUNNEL DRIVE MODE ON!!!")
                drive_mode = "LIDAR"
                tunnel_detect_flag = None
                
        # Find Midpoint to follow & Detect Crosswalk
        if drive_mode == "CAM":
            midpoint, crosswalk_flag = cam_drive.find_midpoint()
            
            # Avoid Obstacle
            if ar_ID is None and cluster_distance < 0.35:                                    
                midpoint += closest_cluster_center[0] * 1000
                
        # Tunnel Mission
        else:
            cam_midpoint, _ = cam_drive.find_midpoint()
            
            midpoint = lidar_drive.find_midpoint()
            drive_type = "TUNNEL DRIVING"
            print("TUNNEL DRIVING!!!")
            
            # Finish tunnel
            _, right_side = lidar_drive.get_side_distance()

            if right_side > 0.8:
                stack += 1
            else:
                stack = 0
            if stack > 10:
                drive_mode = "CAM"
                print("ESCAPE TUNNEL!!!")
                              
        # Handle Crosswalk
        if crosswalk_flag and is_single_color:
            can_we_go = traffic_light.traffic_single()
            
            while can_we_go == 0:
                can_we_go = traffic_light.traffic_single()
                xycar.drive(angle, speed * can_we_go)
                
                RATE.sleep()
                
            is_single_color = False
            print("Single Color Go Go Go!!!")
        
        # Handle Crossroads & Stop Mission
        if ar_ID:
            if ar_ID == 2:
                print("Crossroad!!!")
                prev_ar_ID = 2

            elif ar_ID == 4:
                while True:
                    midpoint, _ = cam_drive.find_midpoint()
                    _, ar_distance = ar_tag.AR_detect()
                
                    if ar_distance < 0.6:
                        xycar.drive(0, 0)
                        is_done = True
                        break
                    
                    angle = xycar.pid(midpoint)
                    xycar.drive(angle, speed)
                    RATE.sleep()

        # Crossroad driving
        while prev_ar_ID == 2:
            can_we_go, direction = traffic_light.traffic_crossroad()
            crossroad_flag = cam_drive.check_crossroad()
            
            if crossroad_flag:                    
                if direction == 'left':
                    print("Let's go left")
                    midpoint, crosswalk_flag = cam_drive.find_midpoint_crosswalk_left()
                elif direction == 'right':
                    print("Let's go right")
                    midpoint, crosswalk_flag = cam_drive.find_midpoint_crosswalk_right()
            else:
                midpoint, crosswalk_flag = cam_drive.find_midpoint()
            
            if crosswalk_flag:
                crossroad_flag = False
                prev_ar_ID = 0
                break

            angle = xycar.pid(midpoint)
            xycar.drive(angle, speed * can_we_go) 
        
        # Stop if done
        if is_done:
            print("Race is done.")
            break
            
        # Drive the car
        angle = xycar.pid(midpoint, drive_type)
        xycar.drive(angle, speed * can_we_go)
