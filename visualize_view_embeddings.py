import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
from pyquaternion import Quaternion
import sys
import os
import argparse
from os.path import join

from collections import OrderedDict
from imageio import imwrite
import imageio
from PIL import Image
from pdb import set_trace

from utils.builder_utils import time_stamped
sys.path.append('../general-utils')
from rot_utils import rotationMatrixToEulerAngles

sys.path.append('../gps-lfd/python')
from gps.agent.bullet.bullet_utils import get_view_embedding, get_two_view_embedding

sys.path.append('/home/max/projects/gps-lfd')
sys.path.append('/home/msieb/projects/gps-lfd')
#from config import Config as Config # Import approriate config
from config_server import Config_Isaac_Server as Config # Import approriate config
conf = Config()

sys.path.append(conf.TCN_PATH)
# from tcn import define_model_depth as define_model # different model architectures - fix at some p$
#from pose_tcn import define_model as define_model # different model architectures - fix at some point bec$
from view_tcn import define_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1, 2, 4"

EXP_DIR = conf.EXP_DIR
EXP_NAME = conf.EXP_NAME
MODE = conf.MODE
MODEL_FOLDER = conf.MODEL_FOLDER
MODEL_NAME = conf.MODEL_NAME
MODEL_PATH = join(EXP_DIR, EXP_NAME, 'trained_models',MODEL_FOLDER, MODEL_NAME)
RGB_PATH = join(EXP_DIR, EXP_NAME, 'videos', MODE)
print("Video path: ", RGB_PATH)
DEPTH_PATH = join(EXP_DIR, EXP_NAME, 'depth', MODE)
#OUTPUT_PATH = '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/rotating_tcn_$
OUTPUT_PATH = join(EXP_DIR, EXP_NAME, 'videos_features', MODEL_FOLDER, MODE)
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default=MODEL_PATH)
parser.add_argument('--experiment-relative-path', type=str, default='tcn-no-depth-sv-full-frame/valid')


T = conf.T
IMAGE_SIZE = conf.IMAGE_SIZE_RESIZED
FPS = conf.FPS
dt = 1/FPS
USE_CUDA = conf.USE_CUDA
SELECTED_SEQS = conf.SELECTED_SEQS
EMBEDDING_VIZ_VIEWS = conf.EMBEDDING_VIZ_VIEWS
ACTION_DIM = conf.ACTION_DIM

print(SELECTED_SEQS)

def resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def load_tcn_model(model_path, use_cuda=False):
    tcn = define_model(use_cuda, action_dim=ACTION_DIM)
    tcn = torch.nn.DataParallel(tcn, device_ids=range(1))

    # tcn = PosNet()

    # Change dict names if model was created with nn.DataParallel
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in state_dict.items():
    #     name = k[7:] # remove module.
    #     new_state_dict[name] = v
    # map_location allows us to load models trained on cuda to cpu.
    tcn.load_state_dict(state_dict)

    if use_cuda:
        tcn = tcn.cuda()
    return tcn

def load_tcn_weights(model_path):
    # Change dict names if model was created with nn.DataParallel
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in state_dict.items():
    #     name = k[7:] # remove module.
    #     new_state_dict[name] = v
    # map_location allows us to load models trained on cuda to cpu.
    tcn.load_state_dict(state_dict)

def main(args):
    # output_folder = join(OUTPUT_PATH, args.experiment_relative_path)
    print("Model path: ", args.model_path)
    output_folder = OUTPUT_PATH
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)     
    
    tcn = load_tcn_model(args.model_path, use_cuda=USE_CUDA)
    # input_folder = join(INPUT_PATH, args.experiment_relative_path)

    logdir = os.path.join('runs', '/'.join(str.split(args.model_path, '/')[:-1]), 'embeddings_viz', time_stamped()) 
    print("logging to {}".format(logdir)) 
    writer = SummaryWriter(logdir)
    image_buffer = []
    label_buffer = []
    feature_buffer = []
    j = 0
    files = [ p for p in os.listdir(RGB_PATH) if p.endswith('.mp4') ]
    files = sorted(files, key=lambda f: int(f.split('_')[0]))
    for file in files:
        if SELECTED_SEQS is not None and file.split('_')[0] not in SELECTED_SEQS:    
            continue
        if EMBEDDING_VIZ_VIEWS is not None and file.split('view')[1].split('.mp4')[0] not in EMBEDDING_VIZ_VIEWS:
            continue
        print("Processing ", file)
        reader = imageio.get_reader(join(RGB_PATH, file))
        reader_depth = imageio.get_reader(join(DEPTH_PATH, file))
        
        embeddings = np.zeros((len(reader), 4))
        embeddings_episode_buffer = []
        #poses = np.load(join(RGB_PATH, file.split('.mp4')[0]+'.npy'))[:, -4:]  
        i = 0
        
        rgb_buffer = []
        depth_buffer = []
        for im, im_depth in zip(reader, reader_depth):
            rgb_buffer.append(im)
            depth_buffer.append(im_depth)

        for i in range(0, len(reader)):
            #i += 1
            #if i % 5 != 0:
            #    continue
            im = rgb_buffer[i]
            im_depth = depth_buffer[i]
            im_before = rgb_buffer[i-1]
            im_depth_before = depth_buffer[i-1]
            image_buffer.append(im)

            resized_image = resize_frame(im, IMAGE_SIZE)[None, :]
            resized_depth = resize_frame(im_depth, IMAGE_SIZE)[None, :]
            # resized_depth = resize_frame(depth_rescaled[:, :, None], IMAGE_SIZE)[None, :]
            frame = np.concatenate([resized_image[0], resized_depth[0, None, 0]], axis=0)
            resized_image_before = resize_frame(im_before, IMAGE_SIZE)[None, :]
            resized_depth_before = resize_frame(im_depth_before, IMAGE_SIZE)[None, :]
            # resized_depth = resize_frame(depth_rescaled[:, :, None], IMAGE_SIZE)[None, :]
            #frames = np.concatenate(resized_image, axis=0)
            #emb_unnormalized, a_pred = get_view_embedding(tcn, resized_image, use_cuda=USE_CUDA)
            emb_unnormalized, a_pred = get_two_view_embedding(tcn, resized_image_before, resized_image, use_cuda=USE_CUDA)

            embedding = emb_unnormalized/ np.linalg.norm(emb_unnormalized)
            embeddings_episode_buffer.append(embedding)
            label_buffer.append(int(file.split('_')[0])) # video sequence label
            #label_buffer.append(poses[i-1]) # video sequence label
            #label_buffer.append(np.concatenate([delta_euler,np.array(int(file.split('view')[1].split('.mp4')[0]))[None]])) 
            #label_buffer.append(i) # view label
        feature_buffer.append(np.array(embeddings_episode_buffer))
        j += 1
        if j >= 50:
            break
    print('generate embedding')
    feature_buffer = np.squeeze(np.array(feature_buffer))
    #feature_buffer = np.array(feature_buffer)
    #features = torch.Tensor(np.reshape(np.array(feature_buffer), [feature_buffer.shape[0]*feature_buffer.shape[1], feature_buffer.shape[2]]))
    features = feature_buffer
    label = torch.Tensor(np.asarray(label_buffer))
    images = torch.Tensor(np.transpose(np.asarray(image_buffer)/255.0, [0, 3, 1, 2]))
    writer.add_embedding(features, metadata=label, label_img=images)
    
    print("=" * 10)

    print('Exit function')
    print("logged to {}".format(logdir)) 

if  __name__ == '__main__':
 
    args = parser.parse_args()
    main(args)
