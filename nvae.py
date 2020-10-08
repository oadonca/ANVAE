# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:06:45 2020

@author: Octavian
"""

import tensorflow as tf
import modules

INIT_W = tf.keras.initializers.VarianceScaling()
tf.compat.v1.disable_eager_execution()

class NVAE():
    def __init__(self, im_size=[32, 32], batch_size = 128, n_channel=1, n_class=None,
                 use_label=False, use_supervise=False, add_noise=False, wd=0,
                 enc_weight=1., gen_weight=1., dis_weight=1.,
                 cat_dis_weight=1., cat_gen_weight=1., cls_weight=1.):
        """
            Args:
                im_size (int or list of length 2): size of input image
                n_channel (int): number of input image channel (1 or 3)
                n_class (int): number of classes
                n_code (int): dimension of code
                use_label (bool): whether incoporate label information
                    in the adversarial regularization or not
                use_supervise (bool): whether supervised training or not
                add_noise (bool): whether add noise to encoder input or not
                wd (float): weight decay
                enc_weight (float): weight of autoencoder loss
                gen_weight (float): weight of latent z generator loss
                dis_weight (float): weight of latent z discriminator loss
                cat_gen_weight (float): weight of label y generator loss
                cat_dis_weight (float): weight of label y discriminator loss
                cls_weight (float): weight of classification loss
        """
        self.levels = [2, 2]
        self._n_channel = n_channel
        self._wd = wd
        self._im_size = im_size
        self.n_class = n_class
        self._enc_w = enc_weight
        self._gen_w = gen_weight
        self._dis_w = dis_weight
        self._cat_dis_w = cat_dis_weight
        self._cat_gen_w = cat_gen_weight
        self._cls_w = cls_weight
        self.layers = {}
        self.is_training = True
        self.batch_size = batch_size
        
    def create_train_input(self):
        self.image = tf.compat.v1.placeholder(
            tf.float32, name='image',
            shape=[None, self._im_size[0], self._im_size[1], self._n_channel])
        self.label = tf.compat.v1.placeholder(
            tf.int64, name='label', shape=[None])
        self.real_distribution = tf.compat.v1.placeholder(
            tf.float32, name='real_distribution', shape=[None, self._im_size[0], self._im_size[1], self._n_channel])
        self.real_y = tf.compat.v1.placeholder(
            tf.int64, name='real_y', shape=[None])
        self.lr = tf.compat.v1.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    
    def create_train_model(self):
        self.is_training = True
        self.create_train_input()
        
        with tf.compat.v1.variable_scope('AE', reuse=tf.compat.v1.AUTO_REUSE):
            self.encoder_in = self.image
            self.layers['encoder_out'] = self.encoder(self.encoder_in)
            
            self.layers['z'] = self.sample_latent(self.layers['encoder_out'])
            
            self.decoder_in = self.layers['z']

            #self.layers['decoder_out'] = self.decoder(self.decoder_in)
            #self.layers['sample_im'] = self.layers['decoder_out']

        # with tf.compat.v1.variable_scope('regularization_z'):
            # fake_in = self.layers['z']
            # real_in = self.real_distribution
            # self.layers['fake_z'] = self.latent_discriminator(fake_in)
            # self.layers['real_z'] = self.latent_discriminator(real_in)
            
    def get_reconstruction_train_op(self):
        with tf.name_scope('reconstruction_train'):
            opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            loss = self.get_reconstruction_loss()
            var_list = tf.trainable_variables(scope='AE')
            # print(var_list)
            grads = tf.gradients(loss, var_list)
            return opt.apply_gradients(zip(grads, var_list))
        
    def get_reconstruction_loss(self):
        with tf.name_scope('reconstruction_loss'):
            p_hat = self.layers['decoder_out']
            p = self.image
            autoencoder_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(p - p_hat), axis=[1,2,3]))
            
            return autoencoder_loss * self._enc_w
            
    def encoder(self, inputs):
        with tf.compat.v1.variable_scope('encoder'):
            self.create_encoder_param()
            outs = []
            # (batch, 32, 32, 1) -> (batch, 15, 15, 32)
            outs.append(modules.encoder_cell(
                inputs, self.eweights['ew1'], self.ebiases['eb1'], self.is_training, dim=32,
                wd=self._wd, name='encoder_FC1', init_w=INIT_W, change_dim=True))
            
            # (batch, 15, 15, 32) -> (batch, 15, 15, 32)
            outs.append(modules.encoder_cell(
                outs[len(outs) - 1], self.eweights['ew2'], self.ebiases['eb2'], self.is_training, dim=32,
                wd=self._wd, name='encoder_FC2', init_w=INIT_W))

            # (batch, 15, 15, 32) -> (batch, 7, 7, 64)
            outs.append(modules.encoder_cell(
                outs[len(outs) - 1], self.eweights['ew3'], self.ebiases['eb3'], self.is_training, dim=64,
                wd=self._wd, name='encoder_FC3', init_w=INIT_W, change_dim=True))
            
            # (batch, 15, 15, 32) -> (batch, 7, 7, 64)
            outs.append(modules.encoder_cell(
                outs[len(outs) - 1], self.eweights['ew4'], self.ebiases['eb4'], self.is_training, dim=64,
                wd=self._wd, name='encoder_FC4', init_w=INIT_W))

            return outs
        
    def decoder(self):
        with tf.compat.v1.variable_scope('encoder'):
            self.create_decoder_param()
            
    def sample_latent(self, encoder_out):
        with tf.compat.v1.variable_scope('sample_latent'):
            self.create_linear_param()
            inputs = self.layers['encoder_out']
            zs = []
            for i in range(len(inputs)):
                output = modules.linear(inputs[i], self.lweights['lw{}'.format(i+1)], self.lbiases['lb{}'.format(i+1)])
                print(output.shape)
                
                mean, logvar = tf.split(output, num_or_size_splits=2, axis=1)
                print(mean.shape)
                print(logvar.shape)
                
                print(tf.exp(logvar * .5).shape)
                
                z = tf.random.normal([self.batch_size, mean.shape[1]]) * tf.exp(logvar * .5) + mean
                print(z.shape)
                zs.append(z)
            
            
            return zs       
        
    def get_latent_space(self):
        return self.layers['z']

    def create_encoder_param(self):
        self.eweights = {
            'ew1': tf.Variable(tf.random.normal([3, 3, 1, 32])),
            'ew2': tf.Variable(tf.random.normal([3, 3, 32, 32])),
            'ew3': tf.Variable(tf.random.normal([3, 3, 32, 64])),
            'ew4': tf.Variable(tf.random.normal([3, 3, 64, 64])),
        }
        self.ebiases = {
            'eb1': tf.Variable(tf.random.normal([32])),
            'eb2': tf.Variable(tf.random.normal([32])),
            'eb3': tf.Variable(tf.random.normal([64])),
            'eb4': tf.Variable(tf.random.normal([64])),
        }
        
    def create_decoder_param(self):
        self.dweights = {
            'dw1': tf.Variable(tf.random.normal([3, 3, 1, 32])),
            'dw2': tf.Variable(tf.random.normal([3, 3, 32, 32])),
            'dw3': tf.Variable(tf.random.normal([3, 3, 32, 64])),
            'dw4': tf.Variable(tf.random.normal([3, 3, 64, 64])),
            'dw4': tf.Variable(tf.random.normal([3, 3, 64, 64])),
            'dw4': tf.Variable(tf.random.normal([3, 3, 64, 64])),
        }
        self.dbiases = {
            'eb1': tf.Variable(tf.random.normal([32])),
            'eb2': tf.Variable(tf.random.normal([32])),
            'eb3': tf.Variable(tf.random.normal([64])),
            'eb4': tf.Variable(tf.random.normal([64])),
        }
        
    def create_linear_param(self):
        self.lweights = {
            'lw1': tf.Variable(tf.random.normal([15*15*32, 2*15*15*32])),
            'lw2': tf.Variable(tf.random.normal([15*15*32, 2*15*15*32])),
            'lw3': tf.Variable(tf.random.normal([7*7*64, 2*7*7*64])),
            'lw4': tf.Variable(tf.random.normal([7*7*64, 2*7*7*64])),
        }
        self.lbiases = {
            'lb1': tf.Variable(tf.random.normal([2*15*15*32])),
            'lb2': tf.Variable(tf.random.normal([2*15*15*32])),
            'lb3': tf.Variable(tf.random.normal([2*7*7*64])),
            'lb4': tf.Variable(tf.random.normal([2*7*7*64])),
        }