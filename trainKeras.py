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

image_batch, label_batch = train_data.next_batch()

features, logits, loc, scale, kl_losses = train_model(image_batch)

print("Shape - Features:")
print(features.shape)

print("Shape - Logits:")
print(logits.shape)

print("Shape - Loc:")
print(loc.shape)

print("Shape - Scale:")
print(scale.shape)

print("Shape - kl_losses:")
print(kl_losses.shape)