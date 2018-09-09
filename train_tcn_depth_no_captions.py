import os
import argparse
import torch
import numpy as np
import pickle
import sys
sys.path.append('./utils')

from torch import optim
from torch import nn
from torch import multiprocessing
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from utils.builders import SingleViewDepthTripletBuilder, MultiViewDepthTripletBuilder
from utils.util import distance, Logger, ensure_folder, collate_fn
from utils.vocabulary import Vocabulary
from tcn import define_model_depth
from ipdb import set_trace
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms

from utils.plot_utils import plot_mean

IMAGE_SIZE = (299, 299)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"

ITERATE_OVER_TRIPLETS = 3

EXP_DIR = '/media/msieb/data/tcn_data/experiments/cube_rotating/'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--model-folder', type=str, default=EXP_DIR + 'trained_models/tcn-depth-sv')
    parser.add_argument('--load-model', type=str, required=False)
    # parser.add_argument('--train-directory', type=str, default='./data/multiview-pouring/train/')
    # parser.add_argument('--validation-directory', type=str, default='./data/multiview-pouring/val/')
    parser.add_argument('--train-directory', type=str, default=EXP_DIR + 'videos/train/')
    parser.add_argument('--train-directory-depth', type=str, default=EXP_DIR + 'depth/train/')

    parser.add_argument('--validation-directory', type=str, default=EXP_DIR + 'videos/valid/')
    parser.add_argument('--validation-directory-depth', type=str, default=EXP_DIR + 'depth/valid/')

    parser.add_argument('--minibatch-size', type=int, default=16)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--model-name', type=str, default='tcn-no-labels-mv')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.001)
    parser.add_argument('--triplets-from-videos', type=int, default=5)
    parser.add_argument('--n-views', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.001, help='weighing factor of language loss to triplet loss')

    # parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    # parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    # parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    # parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    # parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    # parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    # parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=32, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    # parser.add_argument('--num_epochs', type=int, default=5)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument('--learning_rate', type=float, default=0.001)
    return parser.parse_args()

args = get_args()
print(args)
builder = SingleViewDepthTripletBuilder

logger = Logger(args.log_file)

def batch_size(epoch, max_size):
    exponent = epoch // 100
    return min(max(2 ** (exponent), 2), max_size)
validation_builder = builder(args.n_views, args.validation_directory, args.validation_directory_depth, IMAGE_SIZE, args, sample_size=2)
validation_set = [validation_builder.build_set() for i in range(14)]
validation_set = ConcatDataset(validation_set)
del validation_builder

def validate(tcn, use_cuda, args):
    # Run model on validation data and log results
    data_loader = DataLoader(
                    validation_set, 
                    batch_size=32, 
                    shuffle=False, 
                    pin_memory=use_cuda,
                    )
    correct_with_margin = 0
    correct_without_margin = 0
    losses = []
    for minibatch, _ in data_loader:
        # frames = Variable(minibatch, require_grad=False)

        if use_cuda:
            frames = minibatch.cuda()

        anchor_frames = frames[:, 0, :, :, :]
        positive_frames = frames[:, 1, :, :, :]
        negative_frames = frames[:, 2, :, :, :]

        anchor_output, unnormalized, _ = tcn(anchor_frames)
        positive_output, _, _ = tcn(positive_frames)
        negative_output, _, _ = tcn(negative_frames)

        d_positive = distance(anchor_output, positive_output)
        d_negative = distance(anchor_output, negative_output)

        assert(d_positive.size()[0] == minibatch.size()[0])

        correct_with_margin += ((d_positive + args.margin) < d_negative).data.cpu().numpy().sum()
        correct_without_margin += (d_positive < d_negative).data.cpu().numpy().sum()

        loss_triplet = torch.clamp(args.margin + d_positive - d_negative, min=0.0).mean()
        loss = loss_triplet
        losses.append(loss.data.cpu().numpy())
    
    loss = np.mean(losses)
    logger.info('val loss: ',loss)

    message = "Validation score correct with margin {with_margin}/{total} and without margin {without_margin}/{total}".format(
        with_margin=correct_with_margin,
        without_margin=correct_without_margin,
        total=len(validation_set)
    )
    logger.info(message)
    return correct_with_margin, correct_without_margin, loss 

def model_filename(model_name, epoch):
    return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)

def save_model(model, filename, model_folder):
    ensure_folder(model_folder)
    model_path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), model_path)


def build_set(queue, triplet_builder, log):
    while 1:
        datasets = []
        for i in range(3):
            dataset = triplet_builder.build_set()
            datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        # log.info('Created {0} triplets'.format(len(dataset)))
        queue.put(dataset)

def create_model(use_cuda):
    tcn = define_model_depth(use_cuda)
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
    tcn = torch.nn.DataParallel(tcn, device_ids=range(torch.cuda.device_count()))
    triplet_builder = builder(args.n_views, \
        args.train_directory, args.train_directory_depth, IMAGE_SIZE, args, sample_size=5)

    queue = multiprocessing.Queue(1)
    dataset_builder_process = multiprocessing.Process(target=build_set, args=(queue, triplet_builder, logger), daemon=True)
    dataset_builder_process.start()

    optimizer = optim.SGD(tcn.parameters(), lr=args.lr_start, momentum=0.9)
    # This will diminish the learning rate at the milestones.
    # 0.1, 0.01, 0.001
    learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 100], gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    trn_losses_ = []
    val_losses_= []
    val_acc_margin_ = []
    val_acc_no_margin_ = []

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


        for _ in range(0, ITERATE_OVER_TRIPLETS):
            losses = []

            for minibatch, _ in data_loader:
                # frames = Variable(minibatch)
                if use_cuda:
                    frames = minibatch.cuda()
                anchor_frames = frames[:, 0, :, :, :]
                positive_frames = frames[:, 1, :, :, :]
                negative_frames = frames[:, 2, :, :, :]

                anchor_output, unnormalized, _ = tcn(anchor_frames)
                positive_output, _, _ = tcn(positive_frames)
                negative_output, _, _ = tcn(negative_frames)

                d_positive = distance(anchor_output, positive_output)
                d_negative = distance(anchor_output, negative_output)

                loss_triplet = torch.clamp(args.margin + d_positive - d_negative, min=0.0).mean()
                loss = loss_triplet
                losses.append(loss.data.cpu().numpy())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        trn_losses_.append(np.mean(losses))
        logger.info('train loss: ', np.mean(losses))

        if epoch % 1 == 0:
            acc_margin, acc_no_margin, loss = validate(tcn, use_cuda, args)
            val_losses_.append(loss)
            val_acc_margin_.append(acc_margin)
            val_acc_no_margin_.append(acc_no_margin)

        if epoch % args.save_every == 0 and epoch != 0:
            logger.info('Saving model.')
            save_model(tcn, model_filename(args.model_name, epoch), args.model_folder)
        plot_mean(trn_losses_, args.model_folder, 'train_loss')
        plot_mean(val_losses_, args.model_folder, 'validation_loss')
        # plot_mean(train_acc_, args.model_folder, 'train_acc')
        plot_mean(val_acc_margin_, args.model_folder, 'validation_accuracy_margin')
        plot_mean(val_acc_no_margin_, args.model_folder, 'validation_accuracy_no_margin')




if __name__ == '__main__':
    main()
