import argparse
import subprocess as sb
import os
import sys
from os.path import join
import importlib
from ipdb import set_trace

EXP_ROOT_DIR = '/media/hdd/msieb/data/tcn_data/experiments' 

#### Loss functions ####
# The following loss functions can be used: 
# -loss_quat: Loss over normalized quaternions (geodesic)
# -loss_rotation: Loss over 3x3 orthonormal rotation matrices(geodesic),
# -loss_axisangle: takes axis angle representation and converts it to Rot problem (geodesic)
# -loss_euler_reparametrize: predicts 6 values to parametrize the three euler angles (HuberLoss/SmoothL1)
#
# The following builders can be used:
# - SingleViewPoseBuilder: Iterates over one view and loads single images with corresponding ground truth pose of the cropped object
#
#
def main(args):
    model_name = args.model_name
    exp_name = args.exp_name
    run_name = args.run_name
    builder = args.builder
    print("-"*20)
    print("Running experiment: ", exp_name)
    print("-"*20)
    print("Training model: ", model_name)
    print("-"*20)
    
    print("Calling: python train_{}.py --exp-name {} --run-name {} --builder {}".format(model_name, exp_name, run_name, builder))
    print("-"*20)
    sb.call(['python', 'train_{}.py'.format(model_name), '--exp-name', exp_name, '--run-name', run_name, '--builder', builder])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VR Demo and Play")
    parser.add_argument('-m', '--model-name', type=str,required=True, help='model architecture used to train')
    parser.add_argument('-e', '--exp-name', type=str,required=True, help='experiment data to use')
    parser.add_argument('-r', '--run-name', type=str,required=True, help='Folder to store log files of this run under')
    # parser.add_argument('-l', '--loss-fn', dest='loss_fn', action='store_const',const=True, required=True, help='Fwhat loss_fn to use')
    parser.add_argument('-b', '--builder', type=str, required=True, help='what builder to use, e.g. singleview, multiview..')
    args = parser.parse_args()
    main(args)

