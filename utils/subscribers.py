#!/usr/bin/env python
import argparse
import copy
import sys
import time
import inspect

import geometry_msgs
from geometry_msgs.msg import (
                                PoseStamped,
                                Pose,
                                Point,
                                Quaternion,
                                )

from cv_bridge import CvBridge, CvBridgeError
import cv2
import moveit_commander
import moveit_msgs.msg
import numpy as np
from pdb import set_trace
import rospy
import roslib
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

class img_subscriber:
    def __init__(self, topic="/camera/color/image_raw"):
        self.image_sub = rospy.Subscriber(topic, Image, self._callback, queue_size=1)
        self.bridge = CvBridge()
        
    def _callback(self,data):       
        try:
            # tmp self.bridge.imgmsg_to_cv2(data, "bgr8")
            tmp = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")   
            
        except CvBridgeError as e:
            print(e)
        # switch channels from BGR to RGB   
        self.img = tmp.copy()[:,:,::-1]

class array_subscriber:
    def __init__(self, topic="/detector/confidence"):
        self.array_sub = rospy.Subscriber(topic, numpy_msg(Floats), self._callback, queue_size=1)

    def _callback(self, data):
        try:
            tmp = data.data
            tmp2 = data
        except:
            print "could not get confidence subscriber data"
        # self.array = np.array(tmp).reshape([8, 3])
        self.array = tmp.reshape(8,3)