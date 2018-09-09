import argparse
import os
from os.path import join
import functools
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import torch
from torch.utils.data import Dataset, TensorDataset
from torch import Tensor
from torch.autograd import Variable
import logging
from pdb import set_trace
import pickle
import sys
import tensorflow as tf
sys.path.append('utils/')

from utils import util

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

OFFSET = 2
FPS = 30

def main(args):
    extract_videos_and_run_rcnn('/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/' + args.target + '/videos/' + args.mode, \
     '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/' + args.target + '/depth/' + args.mode, \
      frame_size=(480, 640))
    # run_rcnn('/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/ltcn/videos/valid', frame_size=(480, 640))

def extract_videos_and_run_rcnn(rootpath, rootpath_depth, frame_size):
    print(rootpath)

    rcnn = util.RCNN(plot_mode=True)
    for file in os.listdir(rootpath):
        # if 'view1' not in file:
        #     continue
        if not file.endswith('.mp4'):
            continue
        filepath = join(rootpath, file)
        filepath_depth = join(rootpath_depth, file)
        folderpath = join(rootpath, file.split('.mp4')[0])
        folderpath_depth = join(rootpath_depth, file.split('.mp4')[0])
        print("save in", folderpath)

        debugpath = join(rootpath, 'debug', file.split('.mp4')[0])

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        if not os.path.exists(debugpath):
            os.makedirs(debugpath)
        if not os.path.exists(folderpath_depth):
            os.makedirs(folderpath_depth)       

        imageio_video = imageio.read(filepath)
        imageio_video_depth = imageio.read(filepath_depth)

        snap_length = len(imageio_video) 
        frames = np.zeros((snap_length, 3, *frame_size))
        frames_depth = np.zeros((snap_length, 3, *frame_size))
        i = 0
        for frame, frame_depth in zip(imageio_video, imageio_video_depth):
            print("Process frame ", i)
            r, fig = rcnn.get_raw_rcnn_results(frame)
            
            save_name = '{0:05d}'.format(i)
            with open(join(folderpath, save_name + '.pkl'), 'wb') as handle:
                pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)
            plt.imsave(join(folderpath, save_name + '.jpg'), frame)
            plt.imsave(join(folderpath_depth, save_name + '.jpg'), frame_depth)

            if fig is not None:
                canvas = FigureCanvas(fig)
                ax = fig.gca()
                canvas.draw()       # draw the canvas, cache the renderer
                output_img = np.array(fig.canvas.renderer._renderer)
                plt.imsave(join(debugpath, save_name + '.jpg'), output_img)
                plt.close(fig)
            i += 1
        print("="*20)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)


def run_rcnn(rootpath, frame_size):
    print(rootpath)
    rcnn = util.RCNN(plot_mode=True)
    for file in [p for p in os.listdir(rootpath) if not p.endswith('.mp4')]:
        if 'debug' in file:
            continue
        folderpath = join(rootpath, file)
        print("save in", folderpath)

        debugpath = join(rootpath, 'debug', file)

        if not os.path.exists(debugpath):
            os.makedirs(debugpath)

        img_paths = sorted([p for p in os.listdir(folderpath) if p.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
        for i, img_path in enumerate(img_paths):
            save_name = '{0:05d}'.format(i)
            # if os.path.exists(join(folderpath, save_name + '.pkl')):
            #     continue
            print("Process frame ", i)
   
            frame = plt.imread(join(folderpath, img_path))
            r, fig = rcnn.get_raw_rcnn_results(frame)
            with open(join(folderpath, save_name + '.pkl'), 'wb') as handle:
                pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if fig is not None:
                canvas = FigureCanvas(fig)
                ax = fig.gca()
                canvas.draw()       # draw the canvas, cache the renderer
                output_img = np.array(fig.canvas.renderer._renderer)
                plt.imsave(join(debugpath, save_name + '.jpg'), output_img)
                plt.close(fig)
        print("="*20)

def extract_frames():
    INPUT_PATHS = ['videos/' + args.mode, 'depth/' + args.mode]
    for path in INPUT_PATHS:
            OUTDIR='/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/mask_rcnn/mask_estimation_' \
                           + args.mode + '/' + path.split('/')[0] 
            print("saving to {}".format(OUTDIR))
            print("Extracting {}".format(path))
            for file in os.listdir(path):
                if not file.endswith('.mp4'): # or 'view1' in file:
                    continue
                reader = imageio.get_reader(join(path, file))
                dest_folder = join(OUTDIR, file.split('.mp4')[0])
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                print("Extract {}".format(file))
                print("="*10)
                for i, im in enumerate(reader):
                    if i > 0 and not args.mode == 'test':
                        break
                    imageio.imwrite(os.path.join(dest_folder, "{0:05d}.jpg".format(i)), im)
                else:
                    continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('-t', '--target', type=str, required=True)

    args = parser.parse_args()

    main(args)