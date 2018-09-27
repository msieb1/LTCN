import numpy as np
from ipdb import set_trace
import math

def axisAngletoRotationMatrix(a):
    v = a[:-1]
    theta = a[-1]
    R = np.eye(3, 3) 
    R[0, 0] += (-v[2]**2 - v[1]**2)*(1-np.cos(theta)) + 0*np.sin(theta) 
    R[0, 1] += (v[0] * v[1])*(1-np.cos(theta)) - v[2] * np.sin(theta) 
    R[0, 2] += (v[0] * v[2])*(1-np.cos(theta)) + v[1] * np.sin(theta)
    R[1, 0] += (v[0] * v[1])*(1-np.cos(theta)) + v[2] * np.sin(theta)
    R[1, 1] += (-v[2]**2 - v[0]**2)*(1-np.cos(theta)) + 0 * np.sin(theta)
    R[1, 2] += (v[1] * v[2])*(1-np.cos(theta))  - v[0] * np.sin(theta)
    R[2, 0] += (v[0] * v[2])*(1-np.cos(theta))  - v[1] * np.sin(theta)
    R[2, 1] += (v[1] * v[2])*(1-np.cos(theta)) + v[0] * np.sin(theta)
    R[2, 2] += (-v[1]**2 - v[0]**2)*(1-np.cos(theta)) + 0 * np.sin(theta)
    return R


a_ = np.array([.0, 0,1.0 ,math.pi/2.0])
a = np.zeros(4)
a[:-1] = a_[:-1] / np.linalg.norm(a_[:-1])
a[-1] = a_[-1]

R = axisAngletoRotationMatrix(a)
set_trace()
