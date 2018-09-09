import sys
import os
import argparse
from os.path import join

from collections import OrderedDict
from imageio import imwrite
import imageio
import numpy as np
from pygame import mixer
from PIL import Image
import torch

from tcn import define_model

from ipdb import set_trace

EMBEDDING_DIM = 32
T = 100
IMAGE_SIZE = (299, 299)
dt = 0.05

INPUT_PATH = '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/videos'
OUTPUT_PATH = '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/embeddings'

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2"

def resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def load_model(model_path, use_cuda=True):
    tcn = define_model(use_cuda)
    tcn = torch.nn.DataParallel(tcn, device_ids=range(torch.cuda.device_count()))

    # tcn = PosNet()

    # Change dict names if model was created with nn.DataParallel
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove module.
    #     new_state_dict[name] = v
    # map_location allows us to load models trained on cuda to cpu.
    tcn.load_state_dict(state_dict)

    if use_cuda:
        tcn = tcn.cuda()
    return tcn

def save_np_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    np.save(filepath, file)

def save_image_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    imwrite(filepath, file)

def main(args):
    output_folder = join(OUTPUT_PATH, args.experiment_relative_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)     
    
    tcn = load_model(args.model_path)

    input_folder = join(INPUT_PATH, args.experiment_relative_path)
    for file in os.listdir(input_folder):
        print("Processing ", file)
        reader = imageio.get_reader(join(input_folder, file))
        embeddings = np.zeros((len(reader), EMBEDDING_DIM))
        embeddings_normalized = np.zeros((len(reader), EMBEDDING_DIM))

        for i, im in enumerate(reader):
            resized_image = resize_frame(im, IMAGE_SIZE)[None, :]
            output_normalized, output_unnormalized = tcn(torch.Tensor(resized_image).cuda())
            embeddings[i, :] = output_unnormalized.detach().cpu().numpy()
            embeddings_normalized[i, :] = output_normalized.detach().cpu().numpy()
        print("Saving to ", output_folder)
        save_np_file(folder_path=output_folder, name=file.split('.')[0] + '_' + 'emb', file=embeddings)
        save_np_file(folder_path=output_folder, name=file.split('.')[0] + '_' + 'emb_norm',file=embeddings_normalized)
        print("=" * 10)

    print('Exit function')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./trained_models/tcn/pressing_orange_button/tcn-epoch-125.pk')
    parser.add_argument('--experiment-relative-path', type=str, default='pushing_rings/valid')

    args = parser.parse_args()
    main(args)        
