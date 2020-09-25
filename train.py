# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:56:59 2020

@author: Octavian
"""
import tensorflow as tf
import numpy as np
from dataset import dataset
import nvae


TRAIN_PATH = './Data/mnist_train.csv'
TEST_PATH = './Data/mnist_test.csv'
batch_size = 128



train_data = dataset(data_path = TRAIN_PATH, batch_size=batch_size)

train_model = nvae.NVAE()

with tf.compat.v1.Session as sess:
    sess.run(tf.global_variables_initializer())
    batch_data, batch_labels = train_data.next_batch()
    z = sess.run(train_model.get_latent_space(), feed_dict={train_model.image: batch_data, train_model.label: batch_labels})
    print(1)