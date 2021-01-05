# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:41:56 2020

@author: Octavian
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy 
import encoder
import decoder
import discriminator

class ANVAE(tf.keras.Model):
    def __init__(self, latent_spaces, batch_size):
        super(ANVAE, self).__init__()
        
        self.batch_size = batch_size
        self.latent_spaces = 3
        self.level_sizes = [1, 1, 1]
        self.input_s = [32, 32, 1]
        self.latent_channels = 20
        self.h_dim = 1000
        
        self.encoder = encoder.Encoder(self.latent_spaces, self.input_s)
        self.decoder = decoder.Decoder(self.encoder(tf.zeros([self.batch_size, 32, 32, 1]), False), latent_channels=self.latent_channels, level_sizes=self.level_sizes)
        self.discriminator = discriminator.Discriminator(self.latent_spaces, self.input_s, self.h_dim)
        
        self.lr_ae = .0001
        self.lr_dc = .0001
        self.lr_gen = .0001
        
        self.ae_optimizer = tf.keras.optimizers.Adam(self.lr_ae, clipnorm=2)
        self.gen_optimizer = tf.keras.optimizers.Adam(self.lr_gen, clipnorm=2)
        self.dc_optimizer = tf.keras.optimizers.Adam(self.lr_dc, clipnorm=2)
        
        self.ae_loss_weight = 1.
        self.gen_loss_weight = 6.
        self.dc_loss_weight = 6.
        
        self.lastEncVars = []
        self.lastDecVars = []
        self.lastDiscVars = []
        
        self.debugCount = 0
        self.counter = 1
        
        self.log_writer = tf.summary.create_file_writer(logdir='./tf_summary')
        self.step_count = 0
        
    def encode(self, x, debug=False):
        mus = []
        sigmas = []
        features = self.encoder(x, debug)
        
        for z in features:
            flatten = tf.keras.layers.Flatten()(z)
            dim = flatten.shape[-1]
            dense = tf.keras.layers.Dense(dim*2)(flatten)
            mu, sigma = tf.split(dense, 2, 1)
            mus.append(tf.reshape(mu, z.shape))
            sigmas.append(tf.reshape(sigma, z.shape))
        
        return mus, sigmas, features
    
    def dist_encode(self, mus, sigmas):
        
        dists = []
        
        for mu, sigma in zip(mus, sigmas):
            dists.append(tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma))
            
        return dists
    
    def decode(self, zs):
        return self.decoder(zs)
    
    def discriminator_loss(self, real_output, fake_output, weight = 1.0):
        disc_losses = []
        
        for real, fake in zip(real_output, fake_output):
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output))
            disc_losses.append(weight*(tf.reduce_mean(real_loss)+tf.reduce_mean(fake_loss)))
        
        return disc_losses
    
    def autoencoder_loss(self, inputs, recon, weight = 1.0):
        return weight * tf.reduce_mean(tf.square(inputs-recon))
    
    def generator_loss(self, fake, weight = 1.0):
        
        # Should this be tf.zeros_like or tf.ones_like?
        return weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))
    
    def train_step(self, batch_image, batch_label, flush = False):
               
        with tf.GradientTape() as ae_tape, tf.GradientTape() as dc_tape, tf.GradientTape() as gen_tape:
            
            enc_mu, enc_sigma, enc_out = self.encode(batch_image, False)
            enc_dist = self.dist_encode(enc_mu, enc_sigma)
            
            zs = []
            for q_z in enc_dist:
                zs.append(q_z.sample())
                
            dec_out, _ = self.decode(zs)
        
            # Autoencoder - Encoder + Decoder
            
            ae_loss = self.autoencoder_loss(batch_image, dec_out, self.ae_loss_weight)
        
            # Discriminator
            
            real_dists = []
            
            for feature in enc_out:
                real_dists.append(tf.random.normal(shape=feature.shape, mean=0.0, stddev=1.0))
            
            dc_real = self.discriminator(real_dists, False)
            dc_fake = self.discriminator(enc_out, False)
            
            dc_losses = self.discriminator_loss(dc_real, dc_fake, self.dc_loss_weight)
                
            dc_accuracies = []
            
            for real, fake in zip(dc_real, dc_fake):
                dc_accuracies.append(tf.keras.metrics.BinaryAccuracy()(tf.concat([tf.ones_like(real), tf.zeros_like(fake)], axis=0), tf.concat([real, fake], axis=0)))
        
            # Generator - Encoder
            
            gen_loss = self.generator_loss(dc_fake, self.gen_loss_weight)
            
            
        ae_grads = ae_tape.gradient(ae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        
        dc_grads = dc_tape.gradient(dc_losses, self.discriminator.trainable_variables)
        
        gen_grads = gen_tape.gradient(gen_loss, self.encoder.trainable_variables)

        self.ae_optimizer.apply_gradients(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))
                 
        self.dc_optimizer.apply_gradients(zip(dc_grads, self.discriminator.trainable_variables))
        
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.encoder.trainable_variables))
        
        with self.log_writer.as_default():
            tf.summary.scalar(name='Autoencoder_loss', data=ae_loss, step=self.step_count)
            for i, dc_loss in enumerate(dc_losses):
                tf.summary.scalar(name='Discriminator_Loss_{}'.format(i), data=dc_loss, step=self.step_count)
            tf.summary.scalar(name='Generator_Loss', data=gen_loss, step=self.step_count)
            for i, (r_dist, e_dist) in enumerate(zip(real_dists, zs)):
                tf.summary.histogram(name='Real_Distribution_{}'.format(i), data=r_dist, step=self.step_count)
                tf.summary.histogram(name='Encoder_Distribution_{}'.format(i), data=e_dist, step=self.step_count)
        
        self.step_count+=1
        return ae_loss, dc_losses, dc_accuracies, gen_loss

                
                
    
    