#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numpy as np

NUM_CLASSES = 30
use_global_normalization = True


# In[2]:


x_train = np.load('raw_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
x_test = np.load('x_test_mel32.npy', allow_pickle=True)
test_keys = np.load('test_keys.npy', allow_pickle=True)
    
x_train.shape, y_train.shape, x_test.shape, test_keys.shape


# In[4]:


import keras4torch as k4t
import torch
import torch.nn as nn

from dataset import SpeechCommandsDataset
from models import wideresnet

def build_model():
    model = wideresnet(depth=28, widen_factor=10, num_classes=NUM_CLASSES)

    model = k4t.Model(model).build([1, 32, 32])
    
    model.compile(optimizer=torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2), 
                    loss=k4t.losses.CELoss(label_smoothing=0.1),
                    metrics=['acc'], device='cuda')

    return model


# In[5]:


from torch.utils.data import DataLoader
from transforms import *

normal_transform = Compose([crop_or_pad, ToLogMelspectrogram(config='1x32x32')])

if use_global_normalization:
    norm = GlobalNormalization(config='mel32')
    normal_transform = Compose([normal_transform, norm])
    x_test = norm(x_test)

data_aug_transform = Compose([TimeShift(), ChangeAmplitude(), ChangeSpeedAndPitch(), normal_transform])


# In[6]:


from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import MultiStepLR

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

y_proba = np.zeros([len(x_test), NUM_CLASSES]).astype(np.float32)
model_name = 'wideresnet28'

for i, (trn, val) in enumerate(kfold.split(x_train, y_train)):
    print(f'Processing fold {i}:')

    model = build_model()
    lr_scheduler = MultiStepLR(model.trainer.optimizer, milestones=[13, 20, 27, 34], gamma=0.3)

    train_set = SpeechCommandsDataset(x_train[trn], y_train[trn], data_aug_transform)
    val_set = SpeechCommandsDataset(x_train[val], y_train[val], normal_transform, use_cache=True)

    history = model.fit(train_set,
            validation_data=val_set,
            batch_size=96,
            epochs=40,
            callbacks=[
                k4t.callbacks.ModelCheckpoint(f'best_{model_name}_{i}.pt', monitor='val_acc'),
                k4t.callbacks.LRScheduler(lr_scheduler)
            ],
            # num_workers=-1 # uncomment this for multiprocessing
    )
  
    model.load_weights(f'best_{model_name}_{i}.pt')
    print(model.evaluate(x_train[val], y_train[val]))
    y_proba += model.predict(x_test, activation='softmax')

y_proba /= kfold.n_splits
np.save(f'{model_name}_{kfold.n_splits}foldcv_proba.npy', y_proba)


# In[ ]:




