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
    def __init__(self, im_size=[32, 32], n_channel=1, n_class=None,
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
            
            self.layers['z'], self.layers['z_mu'], self.layers['z_std'], self.layers['z_log_std'] = self.sample_latent(self.layers['encoder_out'])
            
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
            out = modules.encoder_cell(
                inputs, self.is_training, dim=8,
                wd=self._wd, name='encoder_FC', init_w=INIT_W)
            return out
        
            
    # def decoder(self):
        
    def sample_latent(self, encoder_out):
        with tf.compat.v1.variable_scope('sample_latent'):
            cnn_out = self.layers['encoder_out']
            
            z_mean = modules.linear(
                out_dim=8, layer_dict=self.layers,
                inputs=cnn_out, init_w=INIT_W, wd=self._wd, name='latent_mean')
            z_std = modules.linear(
                out_dim=8, layer_dict=self.layers, nl=modules.softplus,
                inputs=cnn_out, init_w=INIT_W, wd=self._wd, name='latent_std')
            z_log_std = tf.compat.v1.log(z_std + 1e-8)

            b_size = tf.shape(cnn_out)[0]
            z = modules.tf_sample_diag_guassian(z_mean, z_std, b_size, 8)
            return z, z_mean, z_std, z_log_std
        
    def get_latent_space(self):
        return self.layers['z']
