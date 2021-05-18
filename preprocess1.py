#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import librosa as lb
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm

idx_to_label = 'bed bird cat dog down eight five four go happy house left marvin nine no off on one right seven sheila six stop three tree two up wow yes zero'.split(' ')

NUM_CLASSES = len(idx_to_label)

label_to_idx = {idx_to_label[i]: i for i in range(NUM_CLASSES)}

train_data_path = 'data/train'
test_data_path = 'data/test'


# In[2]:


from sklearn.utils import shuffle

def preprocess_train(pipeline):
    x, y = [], []
    for label in idx_to_label:
        label_dir = f'{train_data_path}/{label}'
        for wav_file in tqdm(os.listdir(label_dir)):
            wav_path = label_dir + f'/{wav_file}'
            wav, _ = lb.load(wav_path, sr=SR)
            x.append(pipeline(wav).astype('float32'))
            y.append(label_to_idx[label])
    x, y = shuffle(np.r_[x], np.r_[y], random_state=7)
    return x, y.astype('int64')

def preprocess_test(pipeline):
    x, keys = [], []
    for wav_file in tqdm(os.listdir(test_data_path)):
        wav_path = f'{test_data_path}/{wav_file}'
        wav, _ = lb.load(wav_path, sr=SR)
        x.append(pipeline(wav).astype('float32'))
        keys.append(wav_file)
    return np.r_[x], np.r_[keys]


# In[3]:


from transforms import *

normal_transform = Compose([crop_or_pad, ToLogMelspectrogram(config='1x32x32')])

data_aug_transform = Compose([
    TimeShift(), ChangeAmplitude(), ChangeSpeedAndPitch(), normal_transform])

x_train, y_train = preprocess_train(lambda x:x)
x_test, test_keys = preprocess_test(normal_transform)

gc.collect()

x_train.shape, y_train.shape, x_test.shape, test_keys.shape


# In[4]:


np.save('raw_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test_mel32.npy', x_test)
np.save('test_keys.npy', test_keys)


# In[ ]:




