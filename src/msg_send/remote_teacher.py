#!/usr/bin/env python

import rospy
from msg_send.msg import my_msg

from std_msgs.msg import String###

def callback(msg): 
    print("1. Name : ", msg.last_name + msg.first_name)
    print("2. ID : ", msg.id_number)
    print("3. Phone Number : ", msg.phone_number)
    msg_content = "\"Good afternoon, " + msg.last_name + " " + msg.first_name + "\""
    pub.publish(msg_content)

rospy.init_node('msg_teacher', anonymous=True)

pub = rospy.Publisher('msg_from_xycar', String, queue_size=10)###

sub = rospy.Subscriber('msg_to_xycar', my_msg, callback)

rospy.spin() 
