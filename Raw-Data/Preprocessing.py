import os
import random
import re
import tensorflow as tf
import tensorflow as tf
import numpy as np
import time
import math
import _pickle as cPickle

plt.rcParams['figure.figsize'] = 14, 4 
sfreq = 128
rfreq = 512
fmin = 4
fmax = 45
baseline_secs = 3

def get_all_purp(root_dir):
    all_mat = []
    for dirName, _, fileList in os.walk(root_dir):
        for fname in fileList:
            #print(fname)
            if '.mat' in fname:
                all_mat.append(dirName + '/' + fname)
            elif '.bdf' in fname:
                all_mat.append(dirName + '/' + fname)
            elif '.dat' in fname:
                all_mat.append(dirName + '/' + fname)
    return all_mat
    
    directory = '/content/drive/My Drive/EMOTION/data_preprocessed_python'
    
    dataset = []
labels = []

for filename in get_all_purp(directory):
     print(filename)
     f_epoch = (cPickle.load(open(filename, 'rb'), encoding='latin1'))['data'][:, :32, sfreq*baseline_secs:]
     f_epoch = np.array(f_epoch)
     y = np.zeros((f_epoch.shape[0],2))
     for vid in range(f_epoch.shape[0]):
       y[vid] = (cPickle.load(open(filename, 'rb'), encoding='latin1'))['labels'][vid,0:2]
     print('shape : ', len(f_epoch),' : ', f_epoch.shape, ' : ',)
     dataset.append(f_epoch)
     labels.append(y)

#f_epoch = f_epoch.reshape(f_epoch.shape[0], f_epoch.shape[2], f_epoch.shape[1])
dataset = np.array(dataset)
labels = np.array(labels)
print('shape : ', len(dataset),' : ', dataset.shape)
print('shape : ', labels.shape)
with open('/content/drive/MyDrive/EMOTION/VQVAE-trans/Trans-checkpoints-py/labels_val_arousal.pkl', 'wb') as filepath:
      pickle.dump(labels, filepath)
all_labels = np.zeros((40*2*32,2))
#all_labels_test = np.zeros(12800)
count = 0
labelll = labels[:]
#labelll_test = (labels[-1]).reshape(1,-1)
for sub in labelll:
  for label in sub:
    for i in range(2):
      all_labels[count] = label
      count += 1
##count = 0
##for sub in labelll_test:
##  for label in sub:
##    for i in range(320):
##      all_labels_test[count] = label
##      count += 1
all_data = dataset.reshape(-1,32,3840)
all_labels = all_labels.reshape(-1,2)
from sklearn.model_selection import train_test_split
## MIXED DATA
X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)
print(f"Train Data: {len(X_train)}")
print(f"Validation Data: {len(X_test)}")
with open('/content/drive/MyDrive/EMOTION/VQVAE-trans/Trans-checkpoints-py/final_video_X_train_30.pkl', 'wb') as filepath:
      pickle.dump(X_train, filepath)
with open('/content/drive/MyDrive/EMOTION/VQVAE-trans/Trans-checkpoints-py/labels_final_video_y_train_30.pkl', 'wb') as filepath:
      pickle.dump(y_train, filepath)
with open('/content/drive/MyDrive/EMOTION/VQVAE-trans/Trans-checkpoints-py/final_video_X_test_30.pkl', 'wb') as filepath:
      pickle.dump(X_test, filepath)
with open('/content/drive/MyDrive/EMOTION/VQVAE-trans/Trans-checkpoints-py/labels_final_video_y_test_30.pkl', 'wb') as filepath:
      pickle.dump(y_test, filepath)
