import os
from os.path import join
import numpy as np
import pybullet as p
import sys
sys.path.append('/home/msieb/projects/general-utils')
from rot_utils import eulerAnglesToRotationMatrix, geodesic_dist_quat, isRotationMatrix
from pdb import set_trace
from pyquaternion import Quaternion
import math
###################### CONFIG ######################

class Config(object):
    # Paths
    EXP_DIR = '/home/max/projects/data/experiments'
    #EXP_DIR = '/media/max/Seagate Backup Plus Drive/experiments'
    GPS_EXP_DIR = '/home/max/projects/gps-lfd/experiments'
    EXP_NAME = 'mug'
    HOME_PATH = "/home/max"
    # MODEL_FOLDER = 'tcn-rgb-mv'
    # MODEL_NAME = 'tcn-no-labels-mv-epoch-100.pk'

    MODEL_FOLDER = 'view-pose' 
    MODEL_FOLDER = 'pose-only'
    MODEL_NAME ='2018-09-15-00-03-24/tcn-epoch-450.pk'

    TCN_PATH = '/home/max/projects/LTCN'
    # Training specific
    USE_CUDA = False
    NUM_VIEWS = 4
    MODE = 'train'

    IMG_H = 299 
    IMG_W = 299
     # EMBEDDING_DIM = 42 # With depth
    EMBEDDING_DIM = 32 # Only RGB

    T = 40
    IMAGE_SIZE_RESIZED = (299, 299)
    FPS = 10

    SELECTED_VIEW = 0 # only record features of one view
    N_PREV_FRAMES = 3

    SEQNAME = '24' # FOR PILQR


class Config_Isaac_Server(Config):
    USE_CUDA = True
    EXP_DIR='/media/hdd/msieb/data/tcn_data/experiments'
    TCN_PATH = "/home/msieb/projects/LTCN"
    #SELECTED_SEQS = None
    SELECTED_SEQS = ['0']
    MODEL_FOLDER = 'tcn_sv'
    EXP_NAME = 'red_cube_stacking'
    #EMBEDDING_VIZ_VIEWS = ['0', '1','5','10']
    EMBEDDING_VIZ_VIEWS = None
    MODEL_NAME = '2018-09-26-13-57-51/tcn-epoch-10.pk'
    MODEL_NAME = '2018-09-28-00-46-39/tcn-epoch-20.pk'
    MODEL_NAME = '2018-09-30-20-17-40/tcn-epoch-60.pk'
    MODEL_NAME = '2018-10-02-09-17-28/tcn-epoch-20.pk'

    SELECTED_VIEW = '0'
    MODE = "train"
    NUM_VIEWS = 5 
    ACTION_DIM =4



###################### CAMERA CONFIG ######################
class Camera_Config(object):
    IMG_H = 160
    IMG_W = 160
    DISTANCE = 1.5
    VIEW_PARAMS = []
    VIEW_PARAMS.append(
        {       'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': -91, 
                'pitch': -48, 
                'roll': 0.0, 
                'upAxisIndex': 2
        }
        )    
    VIEW_PARAMS.append(
        {       'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': 270.6, 
                'pitch': -27, 
                'roll': 50.0, 
                'upAxisIndex': 2,
        }
        )
    VIEW_PARAMS.append(
        {       'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': 170, 
                'pitch':   -31, 
                'roll':0.0, 
                'upAxisIndex': 2
        }
        )
    VIEW_PARAMS.append(
        {       'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': 360, 
                'pitch':   -31, 
                'roll':0.0, 
                'upAxisIndex': 2
        }
        )
    PROJ_PARAMS = {
            'nearPlane':  0.01,
            'farPlane':  100,
            'fov':  8,
            'aspect':  IMG_W / IMG_H,
        }
    ROT_MATRICES = []
    for cam in VIEW_PARAMS:
        euler_angles = [cam['yaw'], cam['pitch'], cam['roll']]
        ROT_MATRICES.append(eulerAnglesToRotationMatrix(euler_angles))  


###################### TRAINING CONFIG ######################
class Multi_Camera_Config(Camera_Config):
    n_cams = 1000

    IMG_H = 160
    IMG_W = 160
    DISTANCE = 1.5
    yaw_low = -220
    yaw_high = 60
    pitch_low = -50
    pitch_high = 0
    roll_low = -120
    roll_high = 120
    VIEW_PARAMS = []
    # yaw_r = np.array([ -78,  -31,   49,   -1,   36, -201, -119,    0,  -60, -159, -219,
    #      -4,   24, -215,   55,   43, -215,   12,  -22,  -24,   30, -118,
    #    -180, -146,   10,   59, -218,  -60, -182,  -45,   -7,  -68, -120,
    #     -93, -141,   47,  -65, -124,  -98, -120, -103, -140,   -5, -105,
    #    -214,  -30,    8, -200,  -90,  -36,   12,  -45,  -62,  -25,   -9,
    #     -84, -166,  -49, -116,  -30,   55,  -68, -100,   15,   48,   37,
    #      59, -100, -146,  -48, -210, -178,  -23,  -94, -133, -194,  -65,
    #     -80,  -79,  -81,  -44,  -29, -199, -135, -185,  -78,   32, -203,
    #    -220,   25,    6, -169, -143, -128,  -21, -170, -200,  -49, -128,
    #     -74]) 
    # pitch_r = np.array([-34, -38, -21, -33, -22, -33, -34, -33, -34, -34, -40, -30, -21,
    #    -24, -32, -27, -40, -27, -35, -35, -38, -28, -30, -33, -31, -25,
    #    -27, -30, -22, -39, -31, -29, -24, -32, -25, -27, -25, -24, -37,
    #    -39, -35, -23, -23, -24, -38, -21, -25, -35, -22, -22, -21, -38,
    #    -38, -35, -26, -26, -26, -22, -27, -29, -32, -28, -29, -21, -26,
    #    -27, -24, -23, -21, -28, -34, -36, -36, -25, -21, -25, -28, -34,
    #    -30, -26, -22, -32, -28, -24, -25, -25, -27, -37, -21, -40, -39,
    #    -33, -23, -28, -37, -23, -36, -35, -36, -25]) 
    # roll_r = np.array([  67,  -63,  109,  -40,   96, -108,  -22,   48,  -92,  109,   20,
    #      65,   26,   52,   43,  -78,   71,   81,  -31,   58,  -24,   -7,
    #     -54,   -3,    9, -100,    0,   32,   36,  -72,   15, -113,  -74,
    #      47,  113,   50,  113,  -23,   15,   91,   66,  113, -110,  -95,
    #       9, -118,   39,  -28, -105, -103, -103, -100,   34,   14, -120,
    #     -33,  -78,  -39,  -38,    6,  -44,   65,   65,  -67,  103,   51,
    #     -65, -104,   98, -117,   79,  108,   95,   33, -106,  119,  -87,
    #     -33,   49, -106,  -75,  -50,  101, -106,  -50,  -42,   97, -115,
    #      89,  -34,  -78,   59,  115,  114,    2,   52,  -15,  -65,  -53,
    # #      45])
    # yaw_r = np.array([ -78,  -31,   49,   -1,   36, -201, -119,    0,  -60, -159, -219,
    #      -4,   24, -215,   55,   43, -215,   12,  -22,  -24,   30, -118,
    #    -180, -146,   10,   59, -218,  -60, -182,  -45,   -7,  -68, -120,
    #     -93, -141,   47,  -65, -124,  -98, -120, -103, -140,   -5, -105,
    #    -214,  -30,    8, -200,  -90,  -36,   12,  -45,  -62,  -25,   -9,
    #     -84, -166,  -49, -116,  -30,   55,  -68, -100,   15,   48,   37,
    #      59, -100, -146,  -48, -210, -178,  -23,  -94, -133, -194,  -65,
    #     -80,  -79,  -81,  -44,  -29, -199, -135, -185,  -78,   32, -203,
    #    -220,   25,    6, -169, -143, -128,  -21, -170, -200,  -49, -128,
    #     -74]) 
    # pitch_r = np.array([-34, -38, -21, -33, -22, -33, -34, -33, -34, -34, -40, -30, -21,
    #    -24, -32, -27, -40, -27, -35, -35, -38, -28, -30, -33, -31, -25,
    #    -27, -30, -22, -39, -31, -29, -24, -32, -25, -27, -25, -24, -37,
    #    -39, -35, -23, -23, -24, -38, -21, -25, -35, -22, -22, -21, -38,
    #    -38, -35, -26, -26, -26, -22, -27, -29, -32, -28, -29, -21, -26,
    #    -27, -24, -23, -21, -28, -34, -36, -36, -25, -21, -25, -28, -34,
    #    -30, -26, -22, -32, -28, -24, -25, -25, -27, -37, -21, -40, -39,
    #    -33, -23, -28, -37, -23, -36, -35, -36, -25]) 
    # roll_r = np.array([  67,  -63,  109,  -40,   96, -108,  -22,   48,  -92,  109,   20,
    #      65,   26,   52,   43,  -78,   71,   81,  -31,   58,  -24,   -7,
    #     -54,   -3,    9, -100,    0,   32,   36,  -72,   15, -113,  -74,
    #      47,  113,   50,  113,  -23,   15,   91,   66,  113, -110,  -95,
    #       9, -118,   39,  -28, -105, -103, -103, -100,   34,   14, -120,
    #     -33,  -78,  -39,  -38,    6,  -44,   65,   65,  -67,  103,   51,
    #     -65, -104,   98, -117,   79,  108,   95,   33, -106,  119,  -87,
    #     -33,   49, -106,  -75,  -50,  101, -106,  -50,  -42,   97, -115,
    #      89,  -34,  -78,   59,  115,  114,    2,   52,  -15,  -65,  -53,
    #      45])
    # yaw_r = np.ones(n_cams,)*-90
    yaw_r =  np.concatenate([np.linspace(yaw_low, yaw_high, n_cams/2),
               np.linspace(yaw_high, yaw_low, n_cams - n_cams/2)])                
    # pitch_r = np.ones(n_cams, ) * -50
    pitch_r =  np.concatenate([np.linspace(pitch_low, pitch_high/2,n_cams/2),
                np.linspace(pitch_high - pitch_high/2, pitch_high, n_cams -n_cams/2)])

    pitch_r = (pitch_high - pitch_low)/2*np.sin(np.linspace(0, 360, n_cams) / 180 * math.pi*4) +(pitch_high + pitch_low) / 2.0
    roll_r = np.zeros(n_cams,)

    for i in range(n_cams):
        yaw = yaw_r[i]
        roll= roll_r[i]
        pitch = pitch_r[i]

        VIEW_PARAMS.append(
            {
                'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': yaw, 
                'pitch': pitch, 
                'roll': roll, 
                'upAxisIndex': 2
            }
            )
    PROJ_PARAMS = {
            'nearPlane':  0.01,
            'farPlane':  100,
            'fov':  8,
            'aspect':  IMG_W / IMG_H,
        }
    print(len(VIEW_PARAMS))
    ROT_MATRICES = []
    EULER_ANGLES = []
    ii = 0
    for cam in VIEW_PARAMS:
        euler_angles = [cam['roll'],cam['pitch'], cam['yaw']]
        EULER_ANGLES.append(np.array(euler_angles))
        rot = eulerAnglesToRotationMatrix(np.array(euler_angles)/180*math.pi)
        ROT_MATRICES.append(rot)
    e1 = np.array([20, 30, 40]) / 180 * math.pi
    e2 =  np.array( [21, 31, 41]) / 180 * math.pi
    e3 = np.array([-21, -31, 41]) / 180 *math.pi
    rots = []
    rots.append(Quaternion(matrix=eulerAnglesToRotationMatrix(e1)).elements) 
    rots.append(Quaternion(matrix=eulerAnglesToRotationMatrix(e2)).elements )
    rots.append(Quaternion(matrix=eulerAnglesToRotationMatrix(e3)).elements)
    d1 = geodesic_dist_quat(rots[0], rots[1], tensor=False) 
    d2 = geodesic_dist_quat(rots[1], rots[2], tensor=False) 

class Training_Config(object):
    TTT = 1


###################### DEMO CONFIG ######################

class Demo_Config(object):
    TRAJECTORY_PICKUP = None

