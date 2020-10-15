# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:24:55 2020

@author: Octavian
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy
import encoder
import decoder

class NVAE(tf.keras.Model):
    def __init__(self, latent_spaces, batch_size):
        super(NVAE, self).__init__()
        
        self.batch_size = batch_size
        self.latent_spaces = 3
        self.level_sizes = [1, 1, 1]
        self.input_s = [32, 32, 1]
        self.latent_channels = 20
        
        self.encoder = encoder.Encoder(self.latent_spaces, self.input_s)
        self.decoder = decoder.Decoder(self.encoder(tf.zeros([1, 32, 32, 1])), latent_channels=self.latent_channels, level_sizes=self.level_sizes)
        
    def call(self, image_batch):
        features = self.encoder(image_batch)
        (logits, loc, scale), kl_losses = self.decoder(features)
        return features, logits, loc, scale, kl_losses