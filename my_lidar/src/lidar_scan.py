# #!/usr/bin/env python

# import rospy
# import time
# from sensor_msgs.msg import LaserScan

# lidar_points = None

# def lidar_callback(data):
#     global lidar_points
#     lidar_points = data.ranges

# rospy.init_node('Lidar', anonymous=True)
# rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)

# while not rospy.is_shutdown():
#     if lidar_points == None:
#         continue

#     rtn = ""
#     for i in range(12):
#         rtn += str(format(lidar_points[i*60],'.2f')) + ", "

#     print(rtn[:-2])
#     time.sleep(0.5)


#!/usr/bin/env python

import rospy
import time
from sensor_msgs.msg import LaserScan

class LidarNode:
    def __init__(self):
        rospy.init_node('Lidar', anonymous=True)
        self.lidar_points = None
        self.subscriber = rospy.Subscriber("/scan", LaserScan, self.lidar_callback, queue_size=1)

    def lidar_callback(self, data):
        self.lidar_points = data.ranges

    def run(self):
        while not rospy.is_shutdown():
            if self.lidar_points is None:
                continue

            rtn = ""
            for i in range(12):
                rtn += str(format(self.lidar_points[i*60], '.2f')) + ", "

            print(rtn[:-2])
            time.sleep(0.5)

if __name__ == '__main__':
    node = LidarNode()
    node.run()
