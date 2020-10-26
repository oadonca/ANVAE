# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:17:17 2020

@author: Octavian
"""
import tensorflow as tf
import numpy as np
from dataset import dataset
import nvaeKeras
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display


TRAIN_PATH = './Data/mnist_train.csv'
TEST_PATH = './Data/mnist_test.csv'
LATENT_SPACES = 3
TRAIN_BUF=60000
BATCH_SIZE=16
TEST_BUF=10000
DIMS = (32,32,1)
N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)
N_EPOCHS = 200

def plot_losses(losses):
    fig, axs =plt.subplots(ncols = 4, nrows = 1, figsize= (16,4))
    for index, latent_loss in enumerate(losses.latent_losss): 
        axs[0].plot(latent_loss.values, label = 'latent_loss_{}'.format(index))
    axs[1].plot(losses.discrim_layer_recon_loss.values, label = 'discrim_layer_recon_loss')
    axs[2].plot(losses.disc_real_loss.values, label = 'disc_real_loss')
    axs[2].plot(losses.disc_fake_loss.values, label = 'disc_fake_loss')
    axs[2].plot(losses.gen_fake_loss.values, label = 'gen_fake_loss')
    axs[3].plot(losses.d_prop.values, label = 'd_prop')

    for ax in axs.flatten():
        ax.legend()
    plt.show()

train_data = dataset(data_path = TRAIN_PATH, batch_size=BATCH_SIZE)

train_model = nvaeKeras.NVAE(LATENT_SPACES, BATCH_SIZE)

losses = pd.DataFrame(columns=[
    'd_prop',
    'latent_losss',
    'discrim_layer_recon_loss',
    'gen_fake_loss',
    'disc_fake_loss',
    'disc_real_loss',
])

for epoch in range(N_EPOCHS):
    loss = []
    iteration = 0
    while train_data.epochs_completed == 0:
        image_batch, label_batch = train_data.next_batch()
        train_model.train(image_batch)
        loss.append(train_model.compute_loss(image_batch))
        iteration += 1
        
        print("Iteration: {}".format(iteration))
        
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    
    print("Epoch: {}".format(epoch))
    
    display.clear_output()
    
    plot_losses(losses)
        
    train_data.setup()
    train_data.shuffle_files()
    

    
