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
batch_size = 4



train_data = dataset(data_path = TRAIN_PATH, batch_size=batch_size)

train_model = nvae.NVAE(batch_size=batch_size)
train_model.create_train_model()

tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())
batch_data, batch_labels = train_data.next_batch()
z = sess.run(train_model.get_latent_space(), feed_dict={train_model.image: batch_data})
    
for i in z:
    print(i.shape)

sess.close()