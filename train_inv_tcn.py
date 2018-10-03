import matplotlib
matplotlib.use('Agg')
import os
from os.path import join
import argparse
import torch
import numpy as np
import pickle
import sys
import datetime
sys.path.append('./utils')

from torch import optim
from torch import nn
from torch import multiprocessing
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from utils.builders import TwoFrameInverseBuilder
from utils.builder_utils import distance, Logger, ensure_folder, collate_fn, time_stamped
from utils.vocabulary import Vocabulary
from inv_tcn import define_model
from ipdb import set_trace
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.plot_utils import plot_mean

sys.path.append('/home/max/projects/gps-lfd') 
sys.path.append('/home/msieb/projects/gps-lfd')
from config import Config_Isaac_Server as Config # Import approriate config
conf = Config()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

IMAGE_SIZE = (299, 299)
ITERATE_OVER_TRIPLETS = 1 
N_PREV_FRAMES = 1

EXP_NAME = conf.EXP_NAME
#EXP_DIR = os.path.join('/home/msieb/data/tcn_data/experiments', EXP_NAME)
#EXP_DIR = os.path.join('/home/msieb/projects/data/tcn_data/experiments', EXP_NAME)
EXP_DIR = conf.EXP_DIR
MODEL_FOLDER = conf.MODEL_FOLDER
USE_CUDA = conf.USE_CUDA
NUM_VIEWS = conf.NUM_VIEWS
SAMPLE_SIZE = 10
TRAIN_SEQS_PER_EPOCH = 15
VAL_SEQS = 5 
SELECTED_VIEW = conf.SELECTED_VIEW

builder = TwoFrameInverseBuilder 
logdir = os.path.join('runs', MODEL_FOLDER, time_stamped()) 
print("logging to {}".format(logdir))
writer = SummaryWriter(logdir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--model-folder', type=str, default=join(EXP_DIR, EXP_NAME,'trained_models', MODEL_FOLDER, time_stamped()))
    parser.add_argument('--load-model', type=str, required=False)
    # parser.add_argument('--train-directory', type=str, default='./data/multiview-pouring/train/')
    # parser.add_argument('--validation-directory', type=str, default='./data/multiview-pouring/val/')
    parser.add_argument('--train-directory', type=str, default=join(EXP_DIR, EXP_NAME, 'videos/train/'))

    parser.add_argument('--validation-directory', type=str, default=join(EXP_DIR, EXP_NAME, 'videos/valid/'))

    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--model-name', type=str, default='tcn')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.01)
    parser.add_argument('--triplets-from-videos', type=int, default=5)
    parser.add_argument('--n-views', type=int, default=SELECTED_VIEW)
    parser.add_argument('--alpha', type=float, default=0.001, help='weighing factor of language loss to triplet loss')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=32, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    return parser.parse_args()

args = get_args()
print(args)

logger = Logger(args.log_file)

def batch_size(epoch, max_size):
    exponent = epoch // 100
    return min(max(2 ** (exponent), 2), max_size)
validation_builder = builder(args.n_views, args.validation_directory, IMAGE_SIZE, args, sample_size=int(SAMPLE_SIZE/2.0))
validation_set = [validation_builder.build_set() for i in range(VAL_SEQS)]
validation_set = ConcatDataset(validation_set)
del validation_builder



def loss_fn(tcn, minibatch, lambd=0.1):

    if USE_CUDA:
       anchor_frames = minibatch[0].cuda()
       anchor_poses = minibatch[1].cuda() 
    
    #print("sampled pose: ", anchor_poses[0].detach().cpu().numpy())
    features_after_gt, features_after_pred, anchor_poses_pred, features_before_gt = tcn(anchor_frames, anchor_poses)
    #print("predicted pose: ", anchor_poses_pred[0].detach().cpu().numpy())
    inv_loss = torch.nn.L1Loss()(anchor_poses_pred, anchor_poses) 
    fwd_loss = torch.nn.L1Loss()(features_after_pred, features_after_gt.detach())
    slowness_loss = torcn.nn.L2Loss()(feature_before_gt, features_after_gt.detach())
    loss = inv_loss + 0.5*fwd_loss + 0.2*slowness_loss
    return loss, fwd_loss, inv_loss

def validate(tcn, use_cuda, n_calls):
    # Run model on validation data and log results
    data_loader = DataLoader(
                    validation_set, 
                    batch_size=16, 
                    shuffle=False, 
                    pin_memory=use_cuda,
                    )
    losses = []
    fwd_losses = []
    inv_losses = []
    for minibatch in data_loader:
        # frames = Variable(minibatch, require_grad=False)
        loss, fwd_loss, inv_loss  = loss_fn(tcn, minibatch)
        losses.append(loss.data.cpu().numpy())
        fwd_losses.append(fwd_loss.data.cpu().numpy())
        inv_losses.append(inv_loss.data.cpu().numpy())
        
    writer.add_scalar('data/valid_fwd_loss', np.mean(fwd_losses), n_calls)
    writer.add_scalar('data/valid_inv_loss', np.mean(inv_losses), n_calls)
    writer.add_scalar('data/valid_loss', np.mean(losses), n_calls)
    n_calls += 1
    loss = np.mean(losses)
    logger.info('val loss: ',loss)

    return loss, n_calls

def model_filename(model_name, epoch):
    return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)

def save_model(model, filename, model_folder):
    ensure_folder(model_folder)
    model_path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), model_path)


def build_set(queue, triplet_builder, log):
    while 1:
        datasets = []
        for i in range(TRAIN_SEQS_PER_EPOCH):
            dataset = triplet_builder.build_set()
            datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        # log.info('Created {0} triplets'.format(len(dataset)))
        queue.put(dataset)

def create_model(use_cuda):
    tcn = define_model(use_cuda)
    # tcn = PosNet()
    if args.load_model:
        model_path = os.path.join(
            args.model_folder,
            args.load_model
        )
        # map_location allows us to load models trained on cuda to cpu.
        tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    if use_cuda:
        tcn = tcn.cuda()
    return tcn


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()

    tcn = create_model(use_cuda)
    
    #dummy_state = Variable(torch.rand(1,2,3,299,299).cuda() )
    #dummy_action = Variable(torch.rand(1,3).cuda())
    #writer.add_graph(tcn, (dummy_state, dummy_action ))

    tcn = torch.nn.DataParallel(tcn, device_ids=range(torch.cuda.device_count()))
    triplet_builder = builder(args.n_views, \
        args.train_directory, IMAGE_SIZE, args, sample_size=SAMPLE_SIZE)

    queue = multiprocessing.Queue(1)
    dataset_builder_process = multiprocessing.Process(target=build_set, args=(queue, triplet_builder, logger), daemon=True)
    dataset_builder_process.start()

    optimizer = optim.SGD(tcn.parameters(), lr=args.lr_start, momentum=0.9)
    # This will diminish the learning rate at the milestones.
    # 0.1, 0.01, 0.001
    learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200,500, 1000], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    trn_losses_ = []
    val_losses_= []

    n_iter = 0
    n_valid_iter = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print("=" * 20)
        logger.info("Starting epoch: {0} learning rate: {1}".format(epoch,
            learning_rate_scheduler.get_lr()))
        learning_rate_scheduler.step()

        dataset = queue.get()
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.minibatch_size, # batch_size(epoch, args.max_minibatch_size),
            shuffle=True,
            pin_memory=use_cuda,
        )
        
        losses = []
        fwd_losses = []
        inv_losses = []
        for _ in range(0, ITERATE_OVER_TRIPLETS):
            for minibatch in data_loader:
                # frames = Variable(minibatch, require_grad=False)
                loss, fwd_loss, inv_loss  = loss_fn(tcn, minibatch)
                
                losses.append(loss.data.cpu().numpy()) 
                fwd_losses.append(fwd_loss.data.cpu().numpy()) 
                inv_losses.append(inv_loss.data.cpu().numpy()) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        writer.add_scalar('data/train_fwd_loss', np.mean(fwd_losses), n_iter)
        writer.add_scalar('data/train_inv_loss', np.mean(inv_losses), n_iter)
        writer.add_scalar('data/train_loss', np.mean(losses), n_iter)
        n_iter += 1  
        trn_losses_.append(np.mean(losses))
        logger.info('train loss: ', np.mean(losses))
        writer.add_image('frame_1', minibatch[0][0], 0)
        writer.add_image('frame_2', minibatch[0][2],0)
        writer.add_image('frame_3', minibatch[0][3],0)
        writer.add_image('frame_4', minibatch[0][4],0)
        writer.add_image('frame_5', minibatch[0][5],0)
        if epoch % 1 == 0:
            loss, n_valid_iter = validate(tcn, use_cuda, n_valid_iter)
            val_losses_.append(loss)

        if epoch % args.save_every == 0 and epoch != 0:
            logger.info('Saving model to {}'.format(args.model_folder)) 
            save_model(tcn, model_filename(args.model_name, epoch), args.model_folder)
        plot_mean(trn_losses_, args.model_folder, 'train_loss')
        plot_mean(val_losses_, args.model_folder, 'validation_loss')
        # plot_mean(train_acc_, args.model_folder, 'train_acc')




if __name__ == '__main__':
    main()
