# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:24:55 2020

@author: Octavian
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy
import encoder

class NVAE(tf.keras.Model):
    def __init__(self, latent_spaces, batch_size):
        super(NVAE, self).__init__()
        
        self.batch_size = 16
        self.latent_spaces = 3
        self.input_s = [32, 32, 1]
        
        self.encoder = encoder.Encoder(self.latent_spaces, self.input_s)
        
    def call(self, image_batch):
        features = self.encoder(image_batch)
        return features