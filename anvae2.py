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
        
        self.lr_ae = .001
        self.lr_dc = .001
        self.lr_gen = .001
        
        self.ae_optimizer = tf.keras.optimizers.Adam(self.lr_ae)
        self.gen_optimizer = tf.keras.optimizers.Adam(self.lr_gen)
        self.dc_optimizer = tf.keras.optimizers.Adam(self.lr_dc)
        
        self.ae_loss_weight = 1
        self.gen_loss_weight = 1
        self.dc_loss_weight = 1
        
        self.lastEncVars = []
        self.lastDecVars = []
        self.lastDiscVars = []
        
        self.debugCount = 0
        self.counter = 1
        
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
    
    def dist_encode(self, x):
        mus, sigmas, _ = self.encode(x, False)
        
        dists = []
        
        for mu, sigma in zip(mus, sigmas):
            dists.append(tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma))
            
        return dists
    
    def decode(self, zs):
        return self.decoder(zs)
    
    def discriminator_loss(self, real_output, fake_output, weight = 1.0):
        disc_losses = []
        
        for real, fake in zip(real_output, fake_output):
            real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real), real)
            fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake), fake)
            disc_losses.append(weight*(real_loss+fake_loss))
        
        return disc_losses
    
    def autoencoder_loss(self, inputs, recon, weight = 1.0):
        return weight * tf.keras.losses.MeanAbsoluteError()(inputs, recon)
    
    def generator_loss(self, fake, weight = 1.0):
        return weight * tf.keras.losses.BinaryCrossentropy()(tf.ones_like(fake), fake)
    
    def train_step(self, batch_image, batch_label):
        
        # Autoencoder - Encoder + Decoder
        
        with tf.GradientTape() as ae_tape:
            dist_enc_out = self.dist_encode(batch_image)
            
            zs = []
            for q_z in dist_enc_out:
                zs.append(q_z.sample())
                
            decoder_out, _ = self.decoder(zs)
            
            ae_loss = self.autoencoder_loss(batch_image, decoder_out, self.ae_loss_weight)
            
        ae_grads = ae_tape.gradient(ae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        
        ae_grads, _ = [(tf.clip_by_global_norm(grad, clip_norm=2)) for grad in ae_grads]
        
        self.ae_optimizer.apply_gradients(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))
        
        # Discriminator
        
        with tf.GradientTape() as dc_tape:
            enc_mu, enc_sigma, enc_out = self.encode(batch_image, False)
            
            real_dists = []
            
            for mu, sigma, feature in zip(enc_mu, enc_sigma, enc_out):
                real_dists.append(tf.random.normal(shape=feature.shape, mean=0.0, stddev=1.0))

            
            dc_real = self.discriminator(real_dists, False)
            dc_fake = self.discriminator(enc_out, False)
            
            dc_losses = self.discriminator_loss(dc_real, dc_fake, self.dc_loss_weight)
                
            dc_accuracies = []
            
            for real, fake in zip(dc_real, dc_fake):
                dc_accuracies.append(tf.keras.metrics.BinaryAccuracy()(tf.concat([tf.ones_like(real), tf.zeros_like(fake)], axis=0), tf.concat([real, fake], axis=0)))
                
        dc_grads = dc_tape.gradient(dc_losses, self.discriminator.trainable_variables)
        
        dc_grads, _ = [(tf.clip_by_global_norm(grad, clip_norm=2)) for grad in dc_grads]
        
        self.dc_optimizer.apply_gradients(zip(dc_grads, self.discriminator.trainable_variables))
        
        # Generator - Encoder
        
        with tf.GradientTape() as gen_tape:
            enc_mu, enc_sigma, enc_out = self.encode(batch_image, False)
            
            dc_fake = self.discriminator(enc_out, False)
            
            gen_loss = self.generator_loss(dc_fake, self.gen_loss_weight)
            
        gen_grads = gen_tape.gradient(gen_loss, self.encoder.trainable_variables)
        
        gen_grads, _ = [(tf.clip_by_global_norm(grad, clip_norm=2)) for grad in gen_grads]
        
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.encoder.trainable_variables))
        
        return ae_loss, dc_losses, dc_accuracies, gen_loss

                
                
    
    