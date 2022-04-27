import logging
import logging
import pickle as pickle

import numpy as np
from Dataset import *
from Model import *
from Trainer import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        dataset_reload = pickle.load(input)

    return dataset_reload


def prepare_data():
    # Loading Data after Wavelet Transform
    all_data = np.load('./all_data_32/upto_32_db4.npy')
    all_data = all_data.reshape(-1, 32, 48, 48, 1)

    # Preparing Labels
    with open('./labels_val_arousal.pkl',
              'rb') as filepath:
        labels = pickle.load(filepath)
    labels = labels[:, :, 1]  # 0 = valence, 1 = arousal
    all_labels = np.zeros((40 * 10 * 32))
    count = 0
    labelll = labels[:]

    for sub in labelll:
        for label in sub:
            for i in range(10):
                all_labels[count] = label
                count += 1

    X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

    print(f"Train Data: {len(X_train)}")
    print(f"Validation Data: {len(X_test)}")

    X_train = X_train.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)

    return X_train, X_test, y_train, y_test


def main():
    # Getting data
    X_train, X_test, y_train, y_test = prepare_data()

    # Collate fn for barch training
    collate_fn = lambda batch: t.stack([t.from_numpy(b) for b in batch], dim=0)

    seconds = 60
    dataset = Dataset_train(seconds, X_train, y_train)
    dataset_test = Dataset_test(seconds, X_test, y_test)

    # sample usage
    # save_object(dataset, '/content/drive/My Drive/EMOTION/loader_60_mix.pkl')
    # dataset = load_object('/content/drive/My Drive/EMOTION/loader_60_mix.pkl')
    # dataset_test = load_object('/content/drive/My Drive/EMOTION/loader_60_mix_test.pkl')

    print("Length of dataset is ", len(dataset))
    print("Length of dataset is ", len(dataset_test))

    train_loader = DataLoader(dataset, batch_size=32, pin_memory=False, shuffle=True)  # collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=32, pin_memory=False, shuffle=False)  # collate_fn=collate_fn)

    logger = logging.getLogger(__name__)

    tconf = TrainerConfig(max_epochs=100, batch_size=32, learning_rate=1e-5)

    # sample model config.
    model_config = {"image_size": 48,
                    "patch_size": 16,
                    "num_classes": 2,
                    "dim": 512,
                    "depth": 6,
                    "heads": 8,
                    "mlp_dim": 1024}

    trainer = Trainer(ViT, model_config, train_loader, len(dataset), test_loader, len(dataset_test), tconf)
    trainer.train()


if __name__ == "__main__":
    main()
