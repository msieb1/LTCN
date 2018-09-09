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
from utils.util import (MultiViewTripletCaptionBuilder, distance, Logger, ensure_folder, collate_fn)
from utils.vocabulary import Vocabulary
from tcn import define_model, EncoderCNN, DecoderRNN
from ipdb import set_trace
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms

from utils.plot_utils import plot_mean

IMAGE_SIZE = (299, 299)

os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"

ITERATE_OVER_TRIPLETS = 5

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/')
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--train-directory', type=str, default='./data/multiview-pouring/train/')
    parser.add_argument('--validation-directory', type=str, default='./data/multiview-pouring/validation/')
    parser.add_argument('--vocab-path', type=str, default='./data/vocab.pkl')    
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--model-name', type=str, default='tcn')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.01)
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
builder = MultiViewTripletBuilder
with open(args.vocab_path, 'rb') as f:
    vocab = pickle.load(f)

logger = Logger(args.log_file)

def batch_size(epoch, max_size):
    exponent = epoch // 100
    return min(max(2 ** (exponent), 2), max_size)

validation_builder = builder(args.n_views, args.validation_directory, IMAGE_SIZE, vocab, args, sample_size=100)
validation_set = [validation_builder.build_set() for i in range(10)]
validation_set = ConcatDataset(validation_set)
del validation_builder

def validate(tcn, decoder, use_cuda, args):
    # Run model on validation data and log results
    data_loader = DataLoader(
                    validation_set, 
                    batch_size=64, 
                    shuffle=False, 
                    pin_memory=use_cuda,
                    collate_fn=collate_fn
                    )
    correct_with_margin = 0
    correct_without_margin = 0
    for minibatch, captions, lengths in data_loader:
        frames = Variable(minibatch, volatile=True)

        if use_cuda:
            frames = frames.cuda()

        anchor_frames = frames[:, 0, :, :, :]
        positive_frames = frames[:, 1, :, :, :]
        negative_frames = frames[:, 2, :, :, :]

        anchor_output, unnormalized = tcn(anchor_frames)
        positive_output, _ = tcn(positive_frames)
        negative_output, _ = tcn(negative_frames)

        d_positive = distance(anchor_output, positive_output)
        d_negative = distance(anchor_output, negative_output)
        caption_outputs = decoder(unnormalized, captions, lengths)

        assert(d_positive.size()[0] == minibatch.size()[0])

        correct_with_margin += ((d_positive + args.margin) < d_negative).data.cpu().numpy().sum()
        correct_without_margin += (d_positive < d_negative).data.cpu().numpy().sum()

    message = "Validation score correct with margin {with_margin}/{total} and without margin {without_margin}/{total}".format(
        with_margin=correct_with_margin,
        without_margin=correct_without_margin,
        total=len(validation_set)
    )
    logger.info(message)

def model_filename(model_name, epoch):
    return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)

def save_model(model, filename, model_folder):
    ensure_folder(model_folder)
    model_path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), model_path)


def build_set(queue, triplet_builder, log):
    while 1:
        datasets = []
        for i in range(5):
            dataset = triplet_builder.build_set()
            datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        log.info('Created {0} triplets'.format(len(dataset)))
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
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), \
        args.num_layers).to(device)
    triplet_builder = builder(args.n_views, \
        args.train_directory, IMAGE_SIZE, vocab, args, sample_size=200)

    queue = multiprocessing.Queue(1)
    dataset_builder_process = multiprocessing.Process(target=build_set, args=(queue, triplet_builder, logger), daemon=True)
    dataset_builder_process.start()

    opt_params = list(tcn.parameters()) + list(decoder.parameters()) + list(encoder.parameters())
    optimizer = optim.SGD(opt_params, lr=args.lr_start, momentum=0.9)
    # This will diminish the learning rate at the milestones.
    # 0.1, 0.01, 0.001
    learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 1000], gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print("=" * 20)
        logger.info("Starting epoch: {0} learning rate: {1}".format(epoch,
            learning_rate_scheduler.get_lr()))
        learning_rate_scheduler.step()

        dataset = queue.get()
        logger.info("Got {0} triplets".format(len(dataset)))
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.minibatch_size, # batch_size(epoch, args.max_minibatch_size),
            shuffle=True,
            pin_memory=use_cuda,
            collate_fn=collate_fn
        )

        if epoch % 10 == 0:
            validate(tcn, decoder, use_cuda, args)
        for _ in range(0, ITERATE_OVER_TRIPLETS):
            losses = []
            for minibatch, captions, lengths in data_loader:
                frames = Variable(minibatch)
                if use_cuda:
                    frames = frames.cuda()
                    captions = captions.to(device)
                anchor_frames = frames[:, 0, :, :, :]
                positive_frames = frames[:, 1, :, :, :]
                negative_frames = frames[:, 2, :, :, :]

                anchor_output, unnormalized = tcn(anchor_frames)
                positive_output, _ = tcn(positive_frames)
                negative_output, _ = tcn(negative_frames)

                d_positive = distance(anchor_output, positive_output)
                d_negative = distance(anchor_output, negative_output)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                # features = encoder(anchor_frames)

                caption_outputs = decoder(unnormalized, captions, lengths)
                loss_triplet = torch.clamp(args.margin + d_positive - d_negative, min=0.0).mean()
                loss_language = criterion(caption_outputs, targets)
                loss = loss_triplet + args.alpha * loss_language
                losses.append(loss.data.cpu().numpy())


                tcn.zero_grad()
                decoder.zero_grad()
                encoder.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info('loss: ', np.mean(losses))
        # Generate an caption from the image
        _, sample_feature = tcn(frames[0,0,:,:,:][None])
        sampled_ids = decoder.sample(sample_feature)
        sampled_ids = sampled_ids[0].cpu().numpy() # (1, max_seq_length) -> (max_seq_length)                sampled_caption = []
        sampled_caption = []
        for word_id in captions[0,:].cpu().numpy():
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
        sentence = ' '.join(sampled_caption)
        print("Target: ", sentence,)
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        print ("Prediction: ",sentence)  

        if epoch % args.save_every == 0 and epoch != 0:
            logger.info('Saving model.')
            save_model(tcn, model_filename(args.model_name, epoch), args.model_folder)
        plot_mean(train_loss_, save_dir, 'train_loss')
        plot_mean(test_loss_, save_dir, 'test_loss')
        plot_mean(train_acc_, save_dir, 'train_acc')
        plot_mean(test_acc_, save_dir, 'test_acc')




if __name__ == '__main__':
    main()
