# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:17:17 2020

@author: Octavian
"""
import tensorflow as tf
import numpy as np
from dataset import dataset
import nvaeKeras


TRAIN_PATH = './Data/mnist_train.csv'
TEST_PATH = './Data/mnist_test.csv'
batch_size = 4
latent_spaces = 3

train_data = dataset(data_path = TRAIN_PATH, batch_size=batch_size)

train_model = nvaeKeras.NVAE(latent_spaces, batch_size)

train_model.compile(run_eagerly=True)
print("^"*80)
image_batch, label_batch = train_data.next_batch()

features, image, kl_losses = train_model(image_batch)

print("Shape - Features:")
for feature in features:
    print(feature.shape)

print("Shape - Image:")
print(image.shape)

print("Shape - kl_losses:")
for kl in kl_losses:
    print(kl)