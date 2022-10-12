import random

import numpy as np
from torch.utils.data import Dataset


class Dataset_train(Dataset):
    def __init__(self, seconds, X_train, y_train):
        self.num_pieces = 5600
        self.num_subs = 3
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def sub(self, num_pieces, num_subs):
        sub_IDs = dict()
        count = 0
        subjects = list(range(num_subs))
        random.seed(4)
        random.shuffle(subjects)
        pieces = list(range(num_pieces))
        random.seed(4)
        random.shuffle(pieces)
        for sub in subjects:
            for i in pieces:
                sub_IDs[count] = str(i) + '_' + str(sub)
                count += 1
        return sub_IDs

    def __getitem__(self, idx):

        image_array = self.X_train[idx]
        image_array = image_array.reshape(1, 48, 48)

        label = self.y_train[int(idx / 32)]

        if label >= 5:
            label = 1
        else:
            label = 0

        return image_array, np.array(label)


class Dataset_test(Dataset):
    def __init__(self, seconds, X_test, y_test):
        self.num_pieces = 5600
        self.num_subs = 1
        self.X_test = X_test
        self.y_test = y_test

    def __len__(self):
        return len(self.X_test)

    def sub(self, num_pieces, num_subs):
        sub_IDs = dict()
        count = 0
        subjects = [4]
        random.seed(4)
        random.shuffle(subjects)
        pieces = list(range(num_pieces))
        random.seed(4)
        random.shuffle(pieces)
        for sub in subjects:
            for i in pieces:
                sub_IDs[count] = str(i) + '_' + str(sub)
                count += 1
        return sub_IDs

    def __getitem__(self, idx):
        image_array = self.X_test[idx]
        image_array = image_array.reshape(1, 48, 48)

        label = self.y_test[int(idx / 32)]

        if label >= 5:
            label = 1
        else:
            label = 0

        return image_array, np.array(label)
