#!/usr/bin/env python

import rospy
from msg_send.msg import my_msg

from std_msgs.msg import String###

def callback(msg):###
    print(msg.data)###

rospy.init_node('msg_student', anonymous=True) 
pub = rospy.Publisher('msg_to_xycar', my_msg)

sub = rospy.Subscriber('msg_from_xycar', String, callback)###

msg = my_msg()
msg.first_name = "Sungmin"
msg.last_name = "Her" 
msg.id_number = 20209876 
msg.phone_number = "010-8950-1010"

rate = rospy.Rate(1) 
while not rospy.is_shutdown():
    pub.publish(msg)
    print("sending message")
    rate.sleep()