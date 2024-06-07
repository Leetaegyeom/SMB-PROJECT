#!/usr/bin/env python
# -*- coding: utf-8 -*-

####################################################################
# 프로그램명 : lidar_a2.py
# 작 성 자 : 자이트론
# 생 성 일 : 2020년 07월 23일
# 본 프로그램은 상업 라이센스에 의해 제공되므로 무단 배포 및 상업적 이용을 금합니다.
####################################################################

import rospy, time
from sensor_msgs.msg import LaserScan
from xycar_motor.msg import xycar_motor

import signal
import sys
import os

def signal_handler(sig, frame):
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

ack_msg = xycar_motor()
ack_publisher = None
ranges_l = []

def callback(data):
    global ranges_l, ack_msg
    ranges_l = data.ranges

def drive(angle):
    global ack_msg, ack_publisher
    ack_msg.speed = 2
    ack_msg.angle = angle
    ack_publisher.publish(ack_msg)

def chk_lidar(angle):
    ok = 0
    for degree in range(42,137):
        if ranges_l[degree] <= 0.3:
            ok += 1
        if ok > 3:
            drive_out()
            break
    if ok <= 3:
        drive(angle)

def stop_msg(delay):
    global ack_msg, ack_publisher
    for stop_cnt in range(delay):
        ack_msg.speed = 0
        ack_msg.angle = 0 
        ack_publisher.publish(ack_msg)
        time.sleep(0.1) # Stop

def back_msg(delay):
    global ack_msg, ack_publisher
    for back_cnt in range(delay):
        ack_msg.angle = 0 
        ack_msg.speed = -2 # Move back
        ack_publisher.publish(ack_msg)
        time.sleep(0.1)

def drive_out():
    global ack_msg, ack_publisher
    stop_msg(10)
    back_msg(20)
    stop_msg(10)
    for right_cnt in range(20):
        chk_lidar(50)
        time.sleep(0.1)

rospy.init_node('sensor_drive', anonymous=True)
rospy.Subscriber('/scan', LaserScan, callback, queue_size = 1)
ack_publisher = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

time.sleep(3) #ready to connect

while not rospy.is_shutdown():
    chk_lidar(0)

