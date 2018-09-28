import torch
import numpy as np
from ipdb import set_trace

def axisAngletoRotationMatrix(a):
    v = a[:-1]
    theta = a[-1]
    r11 = 1+(-v[2]**2 - v[1]**2)*(1-torch.cos(theta)) + 0*torch.sin(theta) 
    r12 = (v[0] * v[1])*(1-torch.cos(theta)) - v[2] * torch.sin(theta) 
    r13 = (v[0] * v[2])*(1-torch.cos(theta)) + v[1] * torch.sin(theta)
    r21 = (v[0] * v[1])*(1-torch.cos(theta)) + v[2] * torch.sin(theta)
    r22 = 1+(-v[2]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r23 = (v[1] * v[2])*(1-torch.cos(theta))  - v[0] * torch.sin(theta)
    r31 = (v[0] * v[2])*(1-torch.cos(theta))  - v[1] * torch.sin(theta)
    r32 = (v[1] * v[2])*(1-torch.cos(theta)) + v[0] * torch.sin(theta)
    r33 = 1+(-v[1]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r1 = torch.cat([r11[None],r12[None],r13[None]])
    r2 = torch.cat([r21[None],r22[None],r23[None]])
    r3 = torch.cat([r31[None],r32[None],r33[None]])
    R = torch.stack((r1, r2, r3), dim=0)
    return R


def geodesic_dist(R1, R2):
    mult = torch.matmul(torch.transpose(R1, dim0=0, dim1=1), R2)
    diagonals = torch.mul(mult, torch.eye(3))
    trace = torch.sum(diagonals)
    dist = torch.acos((trace - 1) / 2.0) # implements geodesic distance of two rotation matrix as loss
    return dist

a1 = torch.Tensor(np.array([1,0,0]))
a1 /= torch.norm(a1)
theta1 = torch.Tensor(np.array([1.5]))
v1 = torch.cat((a1, theta1))
a2 = torch.Tensor(np.array([1,0,0]))
a2 /= torch.norm(a2)
theta2 = torch.Tensor(np.array([1.7]))
v2 = torch.cat((a2, theta2))
R1 = axisAngletoRotationMatrix(v1)
theta = np.array([-1.0])
for i in range(0,10):
	v2[-1] = torch.Tensor(theta)
	R2 = axisAngletoRotationMatrix(v2)
	print(geodesic_dist(R1, R2))
	theta += 0.2
