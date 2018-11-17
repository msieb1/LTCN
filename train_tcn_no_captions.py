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
from utils.builders import SingleViewDepthTripletBuilder, MultiViewDepthTripletBuilder, MultiViewTripletBuilder, SingleViewTripletBuilder
from utils.builder_utils import distance, Logger, ensure_folder, collate_fn, time_stamped
from utils.vocabulary import Vocabulary
from models.tcn import define_model
from ipdb import set_trace
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from shutil import copy2
import importlib

from utils.plot_utils import plot_mean

IMAGE_SIZE = (299, 299)
ITERATE_OVER_TRIPLETS = 1 
SAMPLE_SIZE = 100
BUILDER = SingleViewTripletBuilder
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3"
  
EXP_ROOT_DIR = '/media/hdd/msieb/data/tcn_data/experiments'
sys.path.append(EXP_ROOT_DIR)

class Trainer(object):
    def __init__(self, use_cuda, load_model, model_folder, train_directory, validation_directory, builder, args, multi_gpu=True):
        self.use_cuda = use_cuda
        self.load_model = load_model
        self.model_folder = model_folder
        self.validation_directory = validation_directory
        self.train_directory = train_directory
        self.args = args

        self.builder = builder
        self.logdir = join(model_folder, 'logs')
        self.writer = SummaryWriter(self.logdir)
        self.logger = Logger(self.args.log_file)
        self.itr = 0

        # Create Model
        self.model = self.create_model()
        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        copy2(os.path.realpath(__file__).strip('.pyc') + '.py', self.logdir)
        copy2('/'.join(os.path.realpath(__file__).split('/')[:-1])+ '/models/pose_predictor_euler.py', self.logdir)
        copy2('/'.join(os.path.realpath(__file__).split('/')[:-1])+ '/utils/builder_utils.py', self.logdir)
        copy2('/'.join(os.path.realpath(__file__).split('/')[:-1])+ '/utils/builders.py', self.logdir)
        # Build validation set
        validation_builder = builder(self.args.n_views, validation_directory, IMAGE_SIZE, self.args, sample_size=SAMPLE_SIZE)
        validation_set = [validation_builder.build_set() for i in range(6)]
        validation_set = ConcatDataset(validation_set)
        self.len_validation_set = len(validation_set)
        del validation_builder
        self.validation_loader = DataLoader(
                        validation_set, 
                        batch_size=16, 
                        shuffle=False, 
                        pin_memory=self.use_cuda,
                        )
        self.validation_calls = 0
        # Build Training Set
        self.triplet_builder = builder(self.args.n_views, \
            train_directory, IMAGE_SIZE, self.args, sample_size=SAMPLE_SIZE)
        self.training_queue = multiprocessing.Queue(1)
        dataset_builder_process = multiprocessing.Process(target=self.build_set, args=(self.training_queue, self.triplet_builder, self.logger), daemon=True)
        dataset_builder_process.start()

        # Get Logger
   

        # Model specific setup
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr_start, momentum=0.9)
        # This will diminish the learning rate at the milestones ///// 0.1, 0.01, 0.001 if not using automized scheduler
        self.learning_rate_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # self.criterion = nn.CrossEntropyLoss()

    def train(self):

        trn_losses_ = []
        val_losses_= []
        val_acc_margin_ = []
        val_acc_no_margin_ = []

        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.epochs):
            print("=" * 20)
            self.logger.info("Starting epoch: {0} ".format(epoch))

            dataset = self.training_queue.get()
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self.args.minibatch_size, # batch_size(epoch, self.args.max_minibatch_size),
                shuffle=True,
                pin_memory=self.use_cuda,
            )
            
            train_embedding_features_buffer = []
            train_images_buffer = []
            
            for _ in range(0, ITERATE_OVER_TRIPLETS):
                losses = []

                for minibatch, _ in data_loader:
                    # frames = Variable(minibatch)
                    if self.use_cuda:
                        frames = minibatch.cuda()
                    anchor_frames = frames[:, 0, :, :, :]
                    positive_frames = frames[:, 1, :, :, :]
                    negative_frames = frames[:, 2, :, :, :]
            
                    anchor_output, unnormalized, _ = self.model(anchor_frames)
                    positive_output, _, _ = self.model(positive_frames)
                    negative_output, _, _ = self.model(negative_frames)

                    d_positive = distance(anchor_output, positive_output)
                    d_negative = distance(anchor_output, negative_output)

                    loss_triplet = torch.clamp(self.args.margin + d_positive - d_negative, min=0.0).mean()
                    loss = loss_triplet
                    losses.append(loss.data.cpu().numpy())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Add embeddings
                    train_embedding_features_buffer.append(anchor_output)
                    train_images_buffer.append(anchor_frames)
            print("logging to {}".format(self.logdir))

            self.writer.add_scalar('data/train_triplet_loss', np.mean(losses), self.itr)
            self.itr += 1  
            trn_losses_.append(np.mean(losses))
            self.logger.info('train loss: ', np.mean(losses))
            self.writer.add_image('frame_anchor', minibatch[0][0], 0) 
            self.writer.add_image('frame_positive', minibatch[0][1], 1) 
            self.writer.add_image('frame_negative', minibatch[0][2], 2) 

            # Get embeddings
            features = torch.cat(train_embedding_features_buffer).squeeze_()
            # features = train_embedding_features_buffer.view(train_embedding_features_buffer.shape[0]*train_embedding_features_buffer.shape[1], -1)
            # label = torch.Tensor(np.asarray(label_buffer))
            images = torch.cat(train_images_buffer).squeeze_()#/255.0, [0, 3, 1, 2]
            self.writer.add_embedding(features, label_img=images, global_step=epoch)
            
            if epoch % 1 == 0:
                acc_margin, acc_no_margin, loss  = self.validate()
                self.learning_rate_scheduler.step(loss)
                val_losses_.append(loss)
                val_acc_margin_.append(acc_margin)
                val_acc_no_margin_.append(acc_no_margin)

            if epoch % self.args.save_every == 0 and epoch != 0:
                self.logger.info('Saving model.')
                self.save_model(self.model, self.model_filename(self.args.model_name, epoch), join(self.model_folder, 'weight_files'))
                print("logging to {}".format(self.logdir))

            plot_mean(trn_losses_, self.model_folder, 'train_loss')
            plot_mean(val_losses_, self.model_folder, 'validation_loss')
            # plot_mean(train_acc_, self.args.model_folder, 'train_acc')
            plot_mean(val_acc_margin_, self.model_folder, 'validation_accuracy_margin')
            plot_mean(val_acc_no_margin_, self.model_folder, 'validation_accuracy_no_margin')

    def validate(self):
        # Run model on validation data and log results
        correct_with_margin = 0
        correct_without_margin = 0
        losses = []
        for minibatch, _ in self.validation_loader:
            # frames = Variable(minibatch, require_grad=False)

            if self.use_cuda:
                frames = minibatch.cuda()

            anchor_frames = frames[:, 0, :, :, :]
            positive_frames = frames[:, 1, :, :, :]
            negative_frames = frames[:, 2, :, :, :]

            anchor_output, unnormalized, _ = self.model(anchor_frames)
            positive_output, _, _ = self.model(positive_frames)
            negative_output, _, _ = self.model(negative_frames)
            
            d_positive = distance(anchor_output, positive_output)
            d_negative = distance(anchor_output, negative_output)

            assert(d_positive.size()[0] == minibatch.size()[0])

            correct_with_margin += ((d_positive + self.args.margin) < d_negative).data.cpu().numpy().sum()
            correct_without_margin += (d_positive < d_negative).data.cpu().numpy().sum()

            loss_triplet = torch.clamp(self.args.margin + d_positive - d_negative, min=0.0).mean()
            loss = loss_triplet
            losses.append(loss.data.cpu().numpy())
        self.writer.add_scalar('data/validation_loss', np.mean(losses), self.validation_calls) 
        self.writer.add_scalar('data/validation_correct_with_margin', correct_with_margin / self.len_validation_set, self.validation_calls)
        self.writer.add_scalar('data/validation_correct_without_margin', correct_without_margin / self.len_validation_set, self.validation_calls)
        self.validation_calls += 1
        loss = np.mean(losses)
        self.logger.info('val loss: ',loss)

        message = "Validation score correct with margin {with_margin}/{total} and without margin {without_margin}/{total}".format(
            with_margin=correct_with_margin,
            without_margin=correct_without_margin,
            total=self.len_validation_set
        )
        self.logger.info(message)
        return correct_with_margin, correct_without_margin, loss

    def model_filename(self, model_name, epoch):
        return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)

    def save_model(self, model, filename, model_folder):
        ensure_folder(model_folder)
        model_path = os.path.join(model_folder, filename)
        torch.save(model.state_dict(), model_path)


    def build_set(self, queue, triplet_builder, log):
        while 1:
            datasets = []
            for i in range(3):
                dataset = triplet_builder.build_set()
                datasets.append(dataset)
            dataset = ConcatDataset(datasets)
            # log.info('Created {0} triplets'.format(len(dataset)))
            queue.put(dataset)

    def create_model(self):
        model = define_model(pretrained=True)
        # model = PosNet()
        if self.load_model:
            model_path = os.path.join(
                self.model_folder,
                self.load_model
            )
            # map_location allows us to load models trained on cuda to cpu.
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        if self.use_cuda:
            model = model.cuda()
        return model

    def batch_size(self, epoch, max_size):
        exponent = epoch // 100
        return min(max(2 ** (exponent), 2), max_size)

def main(args):
    # module = importlib.import_module(args.exp_name + '.config')
    # conf = getattr(module, 'Config_Isaac_Server')()
    # EXP_DIR = conf.EXP_DIR
    # MODEL_FOLDER = conf.MODEL_FOLDER


    # GPU Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()

    # Load model
    model_folder = join(EXP_ROOT_DIR, args.exp_name, 'trained_models', args.run_name, time_stamped())
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Get data loader builder and loss function
    builder = getattr(importlib.import_module('utils.builders'), args.builder)

    # Define train and validation directories
    train_directory = join(EXP_ROOT_DIR, args.exp_name, 'videos/train/') 
    validation_directory = join(EXP_ROOT_DIR, args.exp_name, 'videos/valid/') 

    # Copies of executed config
    if not os.path.exists('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/experiments'):
        os.makedirs('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/experiments')
    copy2('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/train_tcn_no_captions.py', model_folder)
    copy2('/'.join(os.path.realpath(__file__).split('/')[:-2]) + '/gps-lfd' + '/config.py', model_folder)
            
    # Build training class
    trainer = Trainer(use_cuda, args.load_model_name, model_folder, train_directory, validation_directory, builder, args) 
    trainer.train()


# def main(args):
#     # GPU Configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     use_cuda = torch.cuda.is_available()
#     builder = BUILDER
    
#     module = importlib.import_module(args.exp_name + '.config')
#     conf = getattr(module, 'Config_Isaac_Server')()
#     model_folder = join(EXP_ROOT_DIR, args.exp_name, 'trained_models', args.run_name, time_stamped())
#     if not os.path.exists(model_folder):
#         os.makedirs(model_folder)
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#     os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1,2,3"

#     EXP_DIR = conf.EXP_DIR
#     MODEL_FOLDER = conf.MODEL_FOLDER
#     train_directory = join(EXP_ROOT_DIR, args.exp_name, 'train/') 
#     validation_directory = join(EXP_ROOT_DIR, args.exp_name, 'valid/') 

#     # Copies of executed config
#     if not os.path.exists('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/experiments'):
#         os.makedirs('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/experiments')
#     copy2('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/train_tcn_no_captions.py', model_folder)
#     copy2('/'.join(os.path.realpath(__file__).split('/')[:-2]) + '/gps-lfd' + '/config.py', model_folder)
            
#     trainer = Trainer(use_cuda, args.load_model, model_folder, train_directory, validation_directory, builder, args) 
#     trainer.train()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--load-model-name', type=str, default='')

    parser.add_argument('--minibatch-size', type=int, default=16)
    parser.add_argument('--margin', type=float, default=3.5)
    parser.add_argument('--model-name', type=str, default='model-weights')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.001)
    parser.add_argument('--triplets-from-videos', type=int, default=5)
    parser.add_argument('--n-views', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.001, help='weighing factor of language loss to triplet loss')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=32, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    # parser.add_argument('--num_epochs', type=int, default=5)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--builder', type=str, required=True)

    args = parser.parse_args()

    main(args)
