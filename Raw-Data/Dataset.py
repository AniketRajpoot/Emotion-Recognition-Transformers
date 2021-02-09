import math
import six
import tensorflow as tf
from tensorflow import keras
import shutil
from enum import Enum
from einops.layers.tensorflow import Rearrange
import logging
import numpy as np
from fastprogress import master_bar, progress_bar
import pandas as pd
import random
import glob
import pickle as pickle
from sklearn.preprocessing import normalize
import _pickle as cPickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import tqdm
from sklearn.model_selection import train_test_split
import fnmatch
import os
import random
import re
import threading
from six.moves import xrange
import time
import json
import torch as t
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import sleep
import math
import functools
from imblearn.over_sampling import RandomOverSampler
import absl.logging as _logging  # pylint: disable=unused-import
import collections
import re
import six
from os.path import join
from six.moves import zip
from absl import flags


class Dataset_train(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, seconds, count):
        'Initialization'

        self.sr = 128
        self.rfreq = 512
        self.fmin = 4
        self.fmax = 45
        self.baseline_secs = 3
        self.seconds = seconds
        self.num_pieces = int(60 / self.seconds)
        self.num_vids = 30
        self.num_subs = 39
        self.num_chans = 32

        with open('./data/final_video_X_train2.pkl',
                  'rb') as filepath:
            self.new_dataset = pickle.load(filepath)
        with open('./data/labels_final_video_y_train2.pkl',
                  'rb') as filepath:
            self.new_labels = pickle.load(filepath)
        self.new_dataset = self.new_dataset.reshape(-1, 768)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.new_dataset)

    def sub(self, num_pieces, num_vids, num_subs, num_chans):
        sub_IDs = dict()
        count = 0
        subjects = list(range(num_subs))
        random.seed(4)
        random.shuffle(subjects)
        videos = list(range(10, num_vids + 10))
        random.seed(4)
        random.shuffle(videos)
        pieces = list(range(num_pieces))
        channels = list(range(num_chans))

        for sub in subjects:
            for vid in videos:
                for i in pieces:
                    for chan in channels:
                        if sub + 1 < 10:
                            if vid < 10:
                                sub_IDs[count] = 's0' + str(sub + 1) + str(i) + '0' + str(vid) + str(chan)
                                count += 1
                            else:
                                sub_IDs[count] = 's0' + str(sub + 1) + str(i) + str(vid) + str(chan)
                                count += 1
                        else:
                            if vid < 10:
                                sub_IDs[count] = 's' + str(sub + 1) + str(i) + '0' + str(vid) + str(chan)
                                count += 1
                            else:
                                sub_IDs[count] = 's' + str(sub + 1) + str(i) + str(vid) + str(chan)
                                count += 1
        return sub_IDs

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID, part, vid, chan = self.sub_IDs[index][:3], int(self.sub_IDs[index][3]), int(self.sub_IDs[index][4:6]), int(self.sub_IDs[index][6:])
        icc = 0
        icc = 1
        # Load data and get label

        X = self.new_dataset[index]
        for indexx, x in enumerate(X):
            X[indexx] = x + 2

        X = X.reshape(-1, 1)

        y = self.new_labels[int(index / 32), 1]

        if y >= 5:
            y = 1.0
        else:
            y = 0.0

        return X, np.array([y])


class Dataset_test(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, seconds, count):
        'Initialization'
        self.sr = 128
        self.rfreq = 512
        self.fmin = 4
        self.fmax = 45
        self.baseline_secs = 3
        self.seconds = seconds
        self.num_pieces = int(60 / self.seconds)
        self.num_vids = 10
        self.num_subs = 32
        self.num_chans = 32

        with open('./data/final_video_X_test2.pkl',
                  'rb') as filepath:
            self.new_dataset = pickle.load(filepath)
        with open('./data/labels_final_video_y_test2.pkl',
                  'rb') as filepath:
            self.new_labels = pickle.load(filepath)
        self.new_dataset = self.new_dataset.reshape(-1, 768)

        self.geneva_ch_names = ['Fp1'
            , 'AF3'
            , 'F3'
            , 'F7'
            , 'FC5'
            , 'FC1'
            , 'C3'
            , 'T7'
            , 'CP5'
            , 'CP1'
            , 'P3'
            , 'P7'
            , 'PO3'
            , 'O1'
            , 'Oz'
            , 'Pz'
            , 'Fp2'
            , 'AF4'
            , 'Fz'
            , 'F4'
            , 'F8'
            , 'FC6'
            , 'FC2'
            , 'Cz'
            , 'C4'
            , 'T8'
            , 'CP6'
            , 'CP2'
            , 'P4'
            , 'P8'
            , 'PO4'
            , 'O2']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.new_dataset)

    def sub(self, num_pieces, num_vids, num_subs, num_chans):
        sub_IDs = dict()
        count = 0
        subjects = list(range(num_subs))
        videos = list(range(num_vids))
        random.shuffle(videos)
        pieces = list(range(num_pieces))
        channels = list(range(num_chans))

        for sub in reversed(subjects):
            for vid in videos:
                for i in pieces:
                    for chan in channels:
                        if sub + 1 < 10:
                            if vid < 10:
                                sub_IDs[count] = 's0' + str(sub + 1) + str(i) + '0' + str(vid) + str(chan)
                                count += 1
                            else:
                                sub_IDs[count] = 's0' + str(sub + 1) + str(i) + str(vid) + str(chan)
                                count += 1
                        else:
                            if vid < 10:
                                sub_IDs[count] = 's' + str(sub + 1) + str(i) + '0' + str(vid) + str(chan)
                                count += 1
                            else:
                                sub_IDs[count] = 's' + str(sub + 1) + str(i) + str(vid) + str(chan)
                                count += 1
        return sub_IDs

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.new_dataset[index]
        for indexx, x in enumerate(X):
            X[indexx] = x + 2

        X = X.reshape(-1, 1)

        y = self.new_labels[int(index / 32), 1]

        if y >= 5:
            y = 1.0
        else:
            y = 0.0

        return X, np.array([y])