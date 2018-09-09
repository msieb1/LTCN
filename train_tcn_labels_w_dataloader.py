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
from utils.util import (MultiViewTripletLabelBuilder, distance, Logger, ensure_folder, collate_fn)
from utils.vocabulary import Vocabulary
from tcn import define_model, EncoderCNN, DecoderRNN, DenseClassifier
from ipdb import set_trace
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
from utils.dataloader import MultiViewTripletLabelDataset
from utils.plot_utils import plot_mean

import random
random.seed(0)
IMAGE_SIZE = (299, 299)

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"

ITERATE_OVER_TRIPLETS = 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--model-folder', type=str, default='/home/msieb/experiments/tcn_data/pushing_rings/trained_models/with_labels_mv')
    parser.add_argument('--load-model', type=str, required=False)
    # parser.add_argument('--train-directory', type=str, default='./data/multiview-pouring/train/')
    # parser.add_argument('--validation-directory', type=str, default='./data/multiview-pouring/val/')
    parser.add_argument('--train-directory', type=str, default='/home/msieb/experiments/tcn_data/pushing_rings/videos/train/')
    parser.add_argument('--validation-directory', type=str, default='/home/msieb/experiments/tcn_data/pushing_rings/videos/valid/')
    parser.add_argument('--labels-train-directory', type=str, default='/home/msieb/experiments/tcn_data/pushing_rings/audio/train/parsed')
    parser.add_argument('--labels-validation-directory', type=str, default='/home/msieb/experiments/tcn_data/pushing_rings/audio/valid/parsed')
    parser.add_argument('--vocab-path', type=str, default='./data/vocab.pkl')    
    parser.add_argument('--minibatch-size', type=int, default=1)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--model-name', type=str, default='tcn-mv')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.01)
    parser.add_argument('--triplets-from-videos', type=int, default=5)
    parser.add_argument('--n-views', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.0005, help='weighing factor of language loss to triplet loss')

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
builder = MultiViewTripletLabelBuilder

validation_set = MultiViewTripletLabelDataset( args.n_views, args.validation_directory, args.labels_validation_directory, IMAGE_SIZE, sample_size=64)
logger = Logger(args.log_file)

def batch_size(epoch, max_size):
    exponent = epoch // 100
    return min(max(2 ** (exponent), 2), max_size)

validation_builder = builder(args.n_views, args.validation_directory, args.labels_validation_directory, IMAGE_SIZE, args, sample_size=100)
# validation_set = [validation_builder.build_set() for i in range(10)]
# validation_set = ConcatDataset(validation_set)
del validation_builder

def validate(tcn, attribute_classifier, criterion, use_cuda, args):
    # Run model on validation data and log results
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(
                    validation_set, 
                    batch_size=1, 
                    shuffle=False, 
                    pin_memory=use_cuda,
                    )
    correct_with_margin = 0
    correct_without_margin = 0
    losses = []
    length_set = 0
    with torch.no_grad():
        correct_1 = 0
        total_1 = 0
        correct_2 = 0
        total_2 = 0

        for frames, caption, seq_idx in data_loader:
            if use_cuda:
                frames = frames[0].to(device)
                captions = caption[0].to(device)  
                seq_idx = seq_idx[0]
            snaps = validation_set.get_videos(int(seq_idx[0]) * args.n_views)
            all_sel_imgs = []
            sel_imgs = snaps[0][::20]
            all_sel_imgs.append(sel_imgs)
            # all_sel_imgs = np.squeeze(np.asarray(all_sel_imgs)).reshape([sel_imgs.shape[0], 3, 299, 299])


            _, unnorm, all_sel_imgs_emb = tcn(torch.FloatTensor(sel_imgs).to(device))
            all_sel_imgs_emb = torch.mean(all_sel_imgs_emb, dim=0, keepdim=True)

            anchor_frames = frames[:, 0, :, :, :]
            positive_frames = frames[:, 1, :, :, :]
            negative_frames = frames[:, 2, :, :, :]

            anchor_output, unnormalized, _ = tcn(anchor_frames)
            positive_output, _, _ = tcn(positive_frames)
            negative_output, _, _ = tcn(negative_frames)

            d_positive = distance(anchor_output, positive_output)
            d_negative = distance(anchor_output, negative_output)
            # features = encoder(anchor_frames)
            loss_triplet = torch.clamp(args.margin + d_positive - d_negative, min=0.0).mean()

            label_outputs_1, label_outputs_2 = attribute_classifier(all_sel_imgs_emb)
            labels_1 = captions[0, 0].view(-1)
            labels_2 = captions[0, 1].view(-1)
            loss_1 = criterion(label_outputs_1, labels_1)
            loss_2 = criterion(label_outputs_2, labels_2) 
            loss_language = loss_1 + loss_2               
            loss = loss_triplet + args.alpha * loss_language
            # loss = loss_language


            losses.append(loss.data.cpu().numpy())

            _, predicted_1 = torch.max(label_outputs_1.data, 1)
            _, predicted_2 = torch.max(label_outputs_2.data, 1)
            total_1 += labels_1.size(0)
            correct_1 += (predicted_1 == labels_1).sum().item()
            total_2 += labels_2.size(0)
            correct_2 += (predicted_2 == labels_2).sum().item()
            length_set += len(anchor_frames)
            print("predicted_1: ", predicted_1)
            print("labels_1: ",labels_1)
            print("predicted_2: ",predicted_2)
            print("labels_2: ", labels_2)
            print('='*10)
        print('Accuracy of active label network branch: {} %'.format(100 * correct_1 / total_1))
        print('Accuracy of passive label network branch: {} %'.format(100 * correct_2 / total_2))
        print("="*10)

        loss = np.mean(losses)
        logger.info('val loss: ',loss)
        message = "Validation score correct with margin {with_margin}/{total} and without margin {without_margin}/{total}".format(
            with_margin=correct_with_margin,
            without_margin=correct_without_margin,
            total=length_set
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
    tcn = torch.nn.DataParallel(tcn, device_ids=(range(torch.cuda.device_count()))) # Wrapper to distribute load on multiple GPUs
    attribute_classifier = DenseClassifier(num_classes=5).to(device) # load labeling network

    # triplet_builder = builder(args.n_views, \
    #     args.train_directory, args.labels_train_directory, IMAGE_SIZE, args, sample_size=200)

    # queue = multiprocessing.Queue(1)
    # dataset_builder_process = multiprocessing.Process(target=build_set, args=(queue, triplet_builder, logger), daemon=True)
    # dataset_builder_process.start()

    optimizer = optim.SGD(list(tcn.parameters()) + list(attribute_classifier.parameters()), lr=args.lr_start, momentum=0.9)
    # This will diminish the learning rate at the milestones.
    # 0.1, 0.01, 0.001
    learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 500], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    trn_losses_ = []
    val_losses_= []
    val_acc_margin_ = []
    val_acc_no_margin_ = []
    dataset = MultiViewTripletLabelDataset( args.n_views, args.train_directory, args.labels_train_directory, IMAGE_SIZE, sample_size=64)

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        losses = []

        print("=" * 20)
        logger.info("Starting epoch: {0} learning rate: {1}".format(epoch,
            learning_rate_scheduler.get_lr()))
        learning_rate_scheduler.step()

        # dataset = queue.get()
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.minibatch_size, # batch_size(epoch, args.max_minibatch_size),
            shuffle=True,
            pin_memory=use_cuda,
        )

        for i, minibatch in enumerate(data_loader):
            frames = minibatch[0]
            caption = minibatch[1]
            seq_idx = minibatch[2]
            if use_cuda:
                frames = frames[0].to(device)
                captions = caption[0].to(device)  
                seq_idx = seq_idx[0]
            snaps = dataset.get_videos(int(seq_idx[0]) * args.n_views)
            all_sel_imgs = []
            sel_imgs = snaps[0][::20]
            all_sel_imgs.append(sel_imgs)
            # all_sel_imgs = np.squeeze(np.asarray(all_sel_imgs)).reshape([sel_imgs.shape[0], 3, 299, 299])

            for _ in range(0, ITERATE_OVER_TRIPLETS):

                _, unnorm, all_sel_imgs_emb = tcn(torch.FloatTensor(sel_imgs).to(device))
                all_sel_imgs_emb = torch.mean(all_sel_imgs_emb, dim=0, keepdim=True)

                anchor_frames = frames[:, 0, :, :, :]
                positive_frames = frames[:, 1, :, :, :]
                negative_frames = frames[:, 2, :, :, :]

                anchor_output, unnormalized, _ = tcn(anchor_frames)
                positive_output, _, _ = tcn(positive_frames)
                negative_output, _, _ = tcn(negative_frames)

                d_positive = distance(anchor_output, positive_output)
                d_negative = distance(anchor_output, negative_output)
                # features = encoder(anchor_frames)
                loss_triplet = torch.clamp(args.margin + d_positive - d_negative, min=0.0).mean()
                if i 
                label_outputs_1, label_outputs_2 = attribute_classifier(all_sel_imgs_emb)
                labels_1 = captions[0, 0].view(-1)
                labels_2 = captions[0, 1].view(-1)
                loss_1 = criterion(label_outputs_1, labels_1)
                loss_2 = criterion(label_outputs_2, labels_2) 
                loss_language = loss_1 + loss_2               
                loss = loss_triplet + args.alpha * loss_language
                # loss = loss_language

                losses.append(loss.data.cpu().numpy())


                tcn.zero_grad()
                attribute_classifier.zero_grad()
                loss.backward()
                optimizer.step()
            if (i+1) % 5 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss Triplet: {:.4f}, Loss Language: {:.4f}' 
                    .format(epoch+1, args.epochs, i+1, len(dataset), loss_triplet.item(), loss_language.item()))


        trn_losses_.append(np.mean(losses))

        if epoch % 1 == 0:
            acc_margin, acc_no_margin, loss = validate(tcn, attribute_classifier, criterion, use_cuda, args)
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
