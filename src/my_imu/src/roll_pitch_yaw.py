#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
import math

last_yaw = None
current_direction = "Straight"
direction_pub = None

def imu_callback(data):
    global last_yaw, current_direction
    orientation_q = data.orientation
    roll, pitch, current_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

    # 로그에 Roll, Pitch, Yaw 출력
    rospy.loginfo("Roll: %.2f, Pitch: %.2f, Yaw: %.2f" % (math.degrees(roll), math.degrees(pitch), math.degrees(current_yaw)))

    if last_yaw is None:
        last_yaw = current_yaw
        return

    # Yaw 변화 계산
    yaw_diff = current_yaw - last_yaw
    yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize yaw_diff to range [-pi, pi]

    # 방향 판단 로직
    if yaw_diff > 0.05:
        current_direction = "Left"
    elif yaw_diff < -0.05:
        current_direction = "Right"
    else:
        current_direction = "Straight"

    last_yaw = current_yaw

def direction_timer_callback(event):
    global current_direction, direction_pub
    direction_pub.publish(String(current_direction))

if __name__ == '__main__':
    rospy.init_node('imu_direction_detector')
    direction_pub = rospy.Publisher('/direction', String, queue_size=1)
    rospy.Subscriber("imu", Imu, imu_callback)
    timer = rospy.Timer(rospy.Duration(1), direction_timer_callback)  # 매초마다 direction_timer_callback 호출

    rospy.spin()
