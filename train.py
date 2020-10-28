# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:17:17 2020

@author: Octavian
"""
import tensorflow as tf
import numpy as np
from dataset import dataset
import anvae
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display

RUN_NAME = "1"
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

def plot_losses(losses, epoch=0):
    fig, axs =plt.subplots(ncols = 4, nrows = 1, figsize= (16,4))
    axs[0].plot(losses.latent_loss1.values, label = 'latent_loss_1')
    axs[0].plot(losses.latent_loss2.values, label = 'latent_loss_2')
    axs[0].plot(losses.latent_loss3.values, label = 'latent_loss_3')
    axs[1].plot(losses.discrim_layer_recon_loss.values, label = 'discrim_layer_recon_loss')
    axs[2].plot(losses.disc_real_loss.values, label = 'disc_real_loss')
    axs[2].plot(losses.disc_fake_loss.values, label = 'disc_fake_loss')
    axs[2].plot(losses.gen_fake_loss.values, label = 'gen_fake_loss')
    axs[3].plot(losses.d_prop.values, label = 'd_prop')

    for ax in axs.flatten():
        ax.legend()
    plt.savefig('Output/{}/plots/losses_plot_{}.png'.format(RUN_NAME, epoch))
    plt.show()
    
def plot_reconstruction(model, data, epoch):
    recon = model.reconstruct(data)
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    for axi, (dat, lab) in enumerate(
        zip(
            [data, recon],
            ["data", "data recon"],
        )
    ):
        for ex in range(1):
            if dat[ex].shape[0] != 32:
                axs[axi].matshow(
                    np.squeeze(dat[ex][0]), cmap=plt.cm.Greys, vmin=0, vmax=1
                )
            else:
                axs[axi].matshow(
                    np.squeeze(dat[ex]), cmap=plt.cm.Greys, vmin=0, vmax=1
                )
            axs[axi].axes.get_xaxis().set_ticks([])
            axs[axi].axes.get_yaxis().set_ticks([])
        axs[axi].set_ylabel(lab)

    plt.savefig('Output/{}/images/image_{}.png'.format(RUN_NAME, epoch))
    plt.show()
    
    
train_data = dataset(data_path = TRAIN_PATH, batch_size=BATCH_SIZE)

train_model = anvae.ANVAE(LATENT_SPACES, BATCH_SIZE)

losses = pd.DataFrame(columns=[
    'd_prop',
    'latent_loss1',
    'latent_loss2',
    'latent_loss3',
    'discrim_layer_recon_loss',
    'gen_fake_loss',
    'disc_fake_loss',
    'disc_real_loss',
])

for epoch in range(N_EPOCHS):
    loss = []
    iteration = 0
    ex_image = None
    while iteration < 2:
        image_batch, label_batch = train_data.next_batch()
        if iteration == 0:
            ex_image=image_batch
        train_model.train(image_batch)
        
        temp2 = []
        
        temp = train_model.compute_loss(image_batch)
        
        for i, l in enumerate(temp):
            if i == 1:
                temp2.append(l[0])
                temp2.append(l[1])
                temp2.append(l[2])
            else:    
                temp2.append(l)
                
        loss.append(temp2)
        iteration += 1

        
        for i, (l, label) in enumerate(zip(loss[iteration-1], losses.columns)):
            print("{}: {}".format(label, l))
        print("Iteration: {}".format(iteration))
        
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    
    print("Epoch: {}".format(epoch))
    
    display.clear_output()
    
    plot_losses(losses, epoch)
    plot_reconstruction(train_model, ex_image, epoch)    
    
    train_data.setup(0, BATCH_SIZE)
    train_data.shuffle_files()
    

    
