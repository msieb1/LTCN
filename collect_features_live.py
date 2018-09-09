import sys
import rospy
import os
import argparse
import imageio

from collections import OrderedDict
from imageio import imwrite
import numpy as np
from pygame import mixer
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from PIL import Image
import torch

from tcn import define_model
from utils.subscribers import img_subscriber
from ipdb import set_trace

EMBEDDING_DIM = 32
T = 100
IMAGE_SIZE = (299, 299)
dt = 0.05
FPS = 1/dt

os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"

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
    rospy.init_node('embeddings_collector', anonymous=True)

    # pub = rospy.Publisher('/tcn/embedding', numpy_msg(Floats), queue_size=3)
    img_subs_obj = img_subscriber(topic="/camera/color/image_raw")
    tcn = load_model(args.model_path)
    try:
        image = img_subs_obj.img

    except:
        print("Camera not working / connected, check connection")
        print("Exit function")
        return

    for i in range(100):
        print('Taking ramp image %d.' % i)
        image = img_subs_obj.img
    len_recording = 1*T
    embeddings = np.zeros((len_recording, EMBEDDING_DIM))
    embeddings_normalized = np.zeros((len_recording, EMBEDDING_DIM))
    image_buffer = []

    mixer.init()
    alert = mixer.Sound('./bell.wav')
    alert.play()
    rospy.sleep(3)
    alert.stop()
    print("="*20)
    print('Starting recording...')
    for t in range(len_recording):
        curr_time = rospy.get_time()
        # print(curr_time)
        image = img_subs_obj.img
        resized_image = resize_frame(image, IMAGE_SIZE)[None, :]
        output_normalized, output_unnormalized = tcn(torch.Tensor(resized_image).cuda())
        embeddings[t, :] = output_unnormalized.detach().cpu().numpy()
        embeddings_normalized[t, :] = output_normalized.detach().cpu().numpy()
        inference_time = rospy.get_time() - curr_time
        image_buffer.append(image)
        rospy.sleep(dt - inference_time)
        # print("runtime: ", rospy.get_time() - curr_time)

    print("Finished recording")
    save_np_file(folder_path=args.experiments_folder, name='emb', file=embeddings)
    save_np_file(folder_path=args.experiments_folder, name='emb_norm',file=embeddings_normalized)

    create_video_of_sample(args.sample_nr, os.path.join(args.experiments_folder, 'videos'), image_buffer)
    # for t, img in enumerate(image_buffer):
    #     save_image_file(folder_path=os.path.join(args.experiments_folder, 'images'), name='{0:05d}.png'.format(t), file=img)
    print('Exit function')



def create_video_of_sample(sample_nr, folder_path, images):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    vidpath = os.path.join(folder_path, 'video_sample_{}.mp4'.format(sample_nr))
    writer = imageio.get_writer(vidpath, fps=FPS)
    for i in range(len(images)):
        writer.append_data(images[i])
    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./trained_models/tcn/pressing_orange_button/tcn-epoch-125.pk')
    parser.add_argument('--experiments-folder', type=str, default='./experiments/pressing_orange_button')
    parser.add_argument('--sample-nr', type=str, default='')

    args = parser.parse_args()
    main(args)        
