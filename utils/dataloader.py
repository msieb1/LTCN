from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from vocabulary import Vocabulary
import nltk
from collections import Counter
import functools
from vocabulary import Vocabulary
import imageio
from PIL import Image

from ipdb import set_trace
OFFSET = 2

def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def view_image(frame):
    # For debugging. Shows the image
    # Input shape (3, 299, 299) float32
    img = Image.fromarray(np.transpose(frame * 255, [1, 2, 0]).astype(np.uint8))
    img.show()

def write_to_csv(values, keys, filepath):
    if  not(os.path.isfile(filepath)):
        with open(filepath, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(keys)
            filewriter.writerow(values)
    else:
        with open(filepath, 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            filewriter.writerow(values)


def ensure_folder(folder):
    path_fragments = os.path.split(folder)
    joined = '.'
    for fragment in path_fragments:
        joined = os.path.join(joined, fragment)
        if not os.path.exists(joined):
            os.mkdir(joined)

def _resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def write_video(file_name, path, frames):
    imageio.mimwrite(os.path.join(path, file_name), frames, fps=60)

def read_video(filepath, frame_size):
    imageio_video = imageio.read(filepath)
    snap_length = len(imageio_video) 
    frames = np.zeros((snap_length, 3, *frame_size))
    resized = map(lambda frame: _resize_frame(frame, frame_size), imageio_video)
    for i, frame in enumerate(resized):
        frames[i, :, :, :] = frame
    return frames

def read_caption(filepath):
    try:
        with open(filepath, 'r') as fp:
            caption = fp.readline()
        return caption
    except:
        print("{} does not exist".format(filepath))
        return None

def read_npy_file(filepath):
    return np.load(filepath)

def ls_directories(path):
    return next(os.walk(path))[1]

# def ls(path):
#     # returns list of files in directory without hidden ones.
#     return sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4] == '.mov')], key=lambda x: int(x.split('_')[0] + x.split('.')[0].split('view')[1]))
#     # randomize retrieval for every epoch?

def ls_npy(path):
    # returns list of files in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p[-4:] == '.npy'], key=lambda x: int(x.split('.')[0]))
    # rand

def ls_txt(path):
    return sorted([p for p in os.listdir(path) if p[0] != '.' and p.endswith('.txt')], key=lambda x: int(x.split('_')[0]))

def ls(path):
    # returns list of files in directory without hidden ones.
    return sorted([p for p in os.listdir(path) if p[0] != '.' and (p[-4:] == '.mp4' or p[-4:] == '.mov')], key=lambda x: int(x.split('_')[0]))
    # rand

class EmbeddingLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, n_views, emb_directory, label_directory):
        self.n_views = n_views
        self._read_embedding_dir(emb_directory) # Creates list of paths with all videos
        self._read_label_dir(label_directory) # Creates list of paths with all videos

        # The negative example has to be from outside the buffer window. Taken from both sides of
        # ihe frame.
        self.sequence_index = 0
        self.vocab = Vocabulary()
        words =['blue', 'orange', 'green', 'blue', 'red', 'yellow']
        for i, word in enumerate(words):
            self.vocab.add_word(word)

    def __len__(self):
        return len(self.emb_paths)

    def __getitem__(self, idx):
        emb = read_npy_file(self.emb_paths[idx])
        seq_idx = self.emb_paths[idx].split('/')[-1].split('_')[0]
        label = read_caption(os.path.join(self._label_directory, seq_idx + '_parsed.txt'))
        label = nltk.tokenize.word_tokenize(str(label).lower())
        active_label = self.vocab(label[-4])
        passive_label = self.vocab(label[-2])
        # Convert caption (string) to word ids.

        target1 = torch.LongTensor([active_label])
        target2 = torch.LongTensor([passive_label])

        emb = np.mean(emb, axis=0)
        emb = torch.FloatTensor(emb)
        return emb, target1, target2

    def _read_embedding_dir(self, emb_directory):
        self._emb_directory = emb_directory
        filenames = ls_npy(emb_directory)
        self.emb_paths = [os.path.join(self._emb_directory, f) for f in filenames]
        self.sequence_count = int(len(self.emb_paths) / self.n_views)

    def _read_label_dir(self, label_directory):
        self._label_directory = label_directory
        filenames = ls_txt(label_directory)
        self.label_paths = [os.path.join(self._label_directory, f) for f in filenames]


class MultiViewTripletLabelDataset(Dataset):

     
    def __init__(self, n_views, video_directory, label_directory, image_size, sample_size=500):
        self.frame_size = image_size
        self.n_views = n_views
        self._read_video_dir(video_directory)
        self.vocab = Vocabulary()
        words =['blue', 'orange', 'green', 'red', 'yellow']
        for i, word in enumerate(words):
            self.vocab.add_word(word)
        self._read_label_dir(label_directory)
        self._count_frames()
        self.sample_size = sample_size
        self.valid_sequence_indices = self._get_valid_sequence_indices(label_directory)
        self.sequence_index = 0
        self.negative_frame_margin = 30
        assert len(self.label_paths) == int(len(self.video_paths) / self.n_views)

    def __len__(self):
        return len(self.valid_sequence_indices)

    def __getitem__(self, idx):
        # build image triplet item
        self.sequence_index = int(idx)
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 3, *self.frame_size)
        label = read_caption(self.label_paths[self.sequence_index])
        label = nltk.tokenize.word_tokenize(str(label).lower())
        # print("index: {}, label: {}".format(self.valid_sequence_indices[idx], label))
        for i in range(self.sample_size):
            snaps = self.get_videos(self.sequence_index * self.n_views)
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snaps)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame    

        try:
            active_label = self.vocab(label[-4])
            passive_label = self.vocab(label[-2])
        except:
            print("Unknown label: ", label)
            print("sequence: ", self.sequence_index)
        seq_idx = torch.LongTensor([self.sequence_index] * self.sample_size)
        # Convert caption (string) to word ids.
        target = torch.LongTensor([[active_label, passive_label]] * self.sample_size) # Needs padded targets of same size as inputs
        return triplets, target, seq_idx

    def _read_video_dir(self, video_directory):
        self._video_directory = video_directory
        filenames = ls(video_directory)
        self.video_paths = [os.path.join(self._video_directory, f) for f in filenames]
        self.sequence_count = int(len(self.video_paths) / self.n_views)

    def _count_frames(self):
        frame_lengths = np.array([len(imageio.read(p)) for p in self.video_paths])
        self.frame_lengths = frame_lengths - OFFSET
        self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
        prev = 0
        for i, frames in enumerate(self.frame_lengths):
            prev = self.cumulative_lengths[i-1]
            self.cumulative_lengths[i] = prev + frames

    def _read_label_dir(self, label_directory):
        self._label_directory = label_directory
        filenames = ls_txt(label_directory)
        self.label_paths = [os.path.join(self._label_directory, f) for f in filenames]   

    def _get_valid_sequence_indices(self, label_directory):
        valid_sequence_indices = []
        curr_seq_idx = 0
        filenames = ls_txt(label_directory)

        for filename in filenames:
            label = read_caption(os.path.join(label_directory, filename))
            label = nltk.tokenize.word_tokenize(str(label).lower())
            if label[-4] is None or label[-2] is None:
                curr_seq_idx += 1
                continue
            else:
                valid_sequence_indices.append(int(filename.split('_')[0]))
                curr_seq_idx += 1

        return valid_sequence_indices

    @functools.lru_cache(maxsize=1)
    def get_videos(self, index):
        views = []
        for i in range(self.n_views):
            views.append(read_video(self.video_paths[index + i], self.frame_size))
        return views

    def sample_triplet(self, snaps):
        loaded_sample = False
        while not loaded_sample:

            try:
                anchor_index = self.sample_anchor_frame_index()
                positive_index = anchor_index
                negative_index = self.sample_negative_frame_index(anchor_index)
                loaded_sample = True
            except:
                print("Error loading video - sequence index: ", self.sequence_index)
                print("video lengths: ", [len(snaps[i]) for i in range(0, len(snaps))])
                print("Maybe margin too high")
        # random sample anchor view,and positive view
        view_set = set(range(self.n_views))
        anchor_view = np.random.choice(np.array(list(view_set)))
        view_set.remove(anchor_view)
        positive_view = np.random.choice(np.array(list(view_set)))
        negative_view = anchor_view # negative example comes from same view INQUIRE TODO

        anchor_frame = snaps[anchor_view][anchor_index]
        positive_frame = snaps[positive_view][positive_index]
        negative_frame = snaps[negative_view][negative_index]
        return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
            torch.Tensor(negative_frame))

    def build_set(self):
        triplets = []
        triplets = torch.Tensor(self.sample_size, 3, 3, *self.frame_size)
        for i in range(0, self.sample_size):
            snaps = self.get_videos(self.sequence_index * self.n_views)
            anchor_frame, positive_frame, negative_frame = self.sample_triplet(snaps)
            triplets[i, 0, :, :, :] = anchor_frame
            triplets[i, 1, :, :, :] = positive_frame
            triplets[i, 2, :, :, :] = negative_frame
        self.sequence_index = (self.sequence_index + 1) % self.sequence_count
        # Second argument is labels. Not used.
        return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

    def sample_anchor_frame_index(self):
        arange = np.arange(0, self.frame_lengths[self.sequence_index * self.n_views])
        return np.random.choice(arange)

    # def sample_positive_frame_index(self, anchor_index):
    #     upper_bound = min(self.frame_lengths[self.sequence_index * self.n_views + 1], anchor_index)
    #     return upper_bound # in case video has less frames than anchor video

    def negative_frame_indices(self, anchor_index):
        video_length = self.frame_lengths[self.sequence_index * self.n_views]
        lower_bound = 0
        upper_bound = max(0, anchor_index - self.negative_frame_margin)
        range1 = np.arange(lower_bound, upper_bound)
        lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
        upper_bound = video_length
        range2 = np.arange(lower_bound, upper_bound)
        return np.concatenate([range1, range2])

    def sample_negative_frame_index(self, anchor_index):
        return np.random.choice(self.negative_frame_indices(anchor_index))