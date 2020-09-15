# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:56:59 2020

@author: Octavian
"""
import tensorflow as tf
import numpy as np
from dataset import dataset


TRAIN_PATH = './Data/mnist_train.csv'
TEST_PATH = './Data/mnist_test.csv'
batch_size = 128



train_data = dataset(data_path = TRAIN_PATH, batch_size=batch_size)

batch_data, batch_labels = train_data.next_batch()

print(batch_data.shape)

