# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:17:17 2020

@author: Octavian
"""
import tensorflow as tf
import numpy as np
from dataset import dataset
import anvae2
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
    axs[0].plot(losses.encoder_loss.values, label = 'encoder_loss')
    axs[0].plot(losses.disc_loss1.values, label = 'disc_loss1')
    axs[0].plot(losses.disc_loss2.values, label = 'disc_loss2')
    axs[1].plot(losses.disc_loss3.values, label = 'disc_loss3')
    axs[2].plot(losses.disc_acc1.values, label = 'disc_acc1')
    axs[2].plot(losses.disc_acc2.values, label = 'disc_acc2')
    axs[2].plot(losses.disc_acc3.values, label = 'disc_acc3')
    axs[3].plot(losses.gen_loss.values, label = 'gen_loss')

    for ax in axs.flatten():
        ax.legend()
    plt.savefig('Output/plots/losses_plot_{}.png'.format(epoch))
    
def plot_reconstruction(model, data, epoch):
    _, _, out = model.encode(data, False)
    recon = model.decode(out)
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    for axi, (dat, lab) in enumerate(
        zip(
            [data, recon],
            ["data", "data recon"],
        )
    ):
        for ex in range(1):
            
            if len(dat[ex].shape) != 3:
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

    plt.savefig('Output/images/image_{}.png'.format(epoch))
    
    
train_data = dataset(data_path = TRAIN_PATH, batch_size=BATCH_SIZE)

train_model = anvae2.ANVAE(LATENT_SPACES, BATCH_SIZE)

losses = pd.DataFrame(columns=[
    'encoder_loss',
    'disc_loss1',
    'disc_loss2',
    'disc_loss3',
    'disc_acc1',
    'disc_acc2',
    'disc_acc3',
    'gen_loss',
])

for epoch in range(N_EPOCHS):
    loss = []
    iteration = 0
    ex_image = None
    epoch_completed = False
    while epoch_completed == False:
        debug = False
        if (iteration % 100 == 0):
            debug = True
        image_batch, label_batch, epoch_completed = train_data.next_batch()
        if iteration == 0:
            ex_image=image_batch
        enc_loss, dc_loss, dc_acc, gen_loss = train_model.train_step(image_batch, label_batch)
        
        temp2 = []
        
        temp2.append(enc_loss)
        for loss in dc_loss:
            temp2.append(loss)
            
        for acc in dc_acc:
            temp2.append(acc)
            
        temp2.append(gen_loss)
                
        losses.append(temp2)

        if (iteration % 100 == 0):
            for i, (l, label) in enumerate(zip(temp2, losses.columns)):
                print("{}: {}".format(label, l))
            print("Iteration: {}".format(iteration))
            print('x'*80)

        if (epoch_completed == True):
            break
        
        iteration += 1
        print(iteration)

    print('1'*80)
        
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    
    print("Epoch: {}".format(epoch))
    
    display.clear_output()
    
    print('2'*80)
    
    plot_losses(losses, epoch)
    
    print('3'*80)
    
    plot_reconstruction(train_model, ex_image, epoch)  
    
    print('4'*80)
    

    
