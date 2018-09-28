
import torch
import numpy as np
import math
from ipdb import set_trace


    # Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 


def norm_sincos(sin, cos):
    stacked_ = torch.cat((sin[None], cos[None]))
    stacked = stacked_ / torch.norm(stacked_)
    return stacked[0], stacked[1]

def sincos2rotm(a_pred):
    # copy of matlab                                                                                        
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx                                                       
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx                                                       
    #        -sy            cy*sx             cy*cx]                                                        
    sinx, cosx = norm_sincos(a_pred[0], a_pred[1]) 
    siny, cosy = norm_sincos(a_pred[2], a_pred[3]) 
    sinz, cosz = norm_sincos(a_pred[4], a_pred[5]) 
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.cat([r11[None],r12[None],r13[None]])
    r2 = torch.cat([r21[None],r22[None],r23[None]])
    r3 = torch.cat([r31[None],r32[None],r33[None]])
    R = torch.stack((r1, r2, r3), dim=0)
    return R 

def axisAngletoRotationMatrix(a):
    v = a[:-1]
    theta = a[-1]
    r11 = 1 + (-v[2]**2 - v[1]**2)*(1-torch.cos(theta)) + 0*torch.sin(theta) 
    r12 =  (v[0] * v[1])*(1-torch.cos(theta)) - v[2] * torch.sin(theta) 
    r13 =  (v[0] * v[2])*(1-torch.cos(theta)) + v[1] * torch.sin(theta)
    r21 =  (v[0] * v[1])*(1-torch.cos(theta)) + v[2] * torch.sin(theta)
    r22 = 1 + (-v[2]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r23 =  (v[1] * v[2])*(1-torch.cos(theta))  - v[0] * torch.sin(theta)
    r31 =  (v[0] * v[2])*(1-torch.cos(theta))  - v[1] * torch.sin(theta)
    r32 =  (v[1] * v[2])*(1-torch.cos(theta)) + v[0] * torch.sin(theta)
    r33 = 1 + (-v[1]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r1 = torch.cat([r11[None],r12[None],r13[None]])
    r2 = torch.cat([r21[None],r22[None],r23[None]])
    r3 = torch.cat([r31[None],r32[None],r33[None]])
    R = torch.stack((r1, r2, r3), dim=0)

    return R



# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta, tensor=False) :
    """
    Theta is given as euler angles Z-Y-X, corresponding to yaw, pitch, roll
    """ 
    if not tensor:
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
             
                            
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])          
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
                                     
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R


 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# Return X-Y-Z (roll pitch yaw)
def rotationMatrixToEulerAngles(R) :
    
    if not R.type() == 'torch.cuda.FloatTensor':
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
         
        singular = sy < 1e-6
     
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])
    else:
        sy = torch.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
     
        if  not singular :
            x = torch.atan2(R[2,1] , R[2,2])
            y = torch.atan2(-R[2,0], sy)
            z = torch.atan2(R[1,0], R[0,0])
        else :
            x = torch.atan2(-R[1,2], R[1,1])
            y = torch.atan2(-R[2,0], sy)
            z = 0
        return torch.stack((x, y, z))

# def create_random_rot(tensor=False):
#       """
#     vector should be 6 dimensional
#     """
#     # random unit vectors
#     u = np.random.rand(3)
#     v = np.random.rand(3)
#     u /= np.linalg.norm(u)
#     v /= np.linalg.norm(v)
#     # subtract (v*u)u from v and normalize
#     v -= v.dot(u)*u
#     v /= np.linalg.norm(v)
#     # build cross product
#     w = np.cross(u, v)
#     w /= np.linalg.norm(w)
#     R = np.hstack([u[:,None], v[:,None], w[:,None]])

#     if tensor:
#         return torch.Tensor(R)
#     else:
#         return R



def create_rot_from_vector(vector):
    """
    vector should be 6 dimensional
    """
    # random unit vectors
    u = vector[:3]
    v = vector[3:]
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    # subtract (v*u)u from v and normalize
    v -= v.dot(u)*u
    v /= np.linalg.norm(v)
    # build cross product
    w = np.cross(u, v)
    w /= np.linalg.norm(w)
    R = np.hstack([u[:,None], v[:,None], w[:,None]])
    return R


