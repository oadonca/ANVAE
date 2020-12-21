# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 06:39:57 2020

@author: Octavian
"""

import tensorflow as tf
import tensorflow_probability as tfp
import modules

class DiscriminatorCell(tf.keras.Model):
    def __init__(self, h_dim, input_s):
        super(DiscriminatorCell, self).__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_s),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(h_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(h_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1)
            ]
        )
        
    def call(self, x):
        return self.seq(x)

class Discriminator(tf.keras.Model):
    def __init__(self, latent_spaces, input_shape, h_dim):
        super(Discriminator, self).__init__()
        self.h_dim = h_dim
        self.z_dim = input_shape[2]
        self.input_s = input_shape
        self.levels = []
        for level in range(latent_spaces):
            
            if(level == 0):
                self.z_dim *= 4
                self.input_s = [self.input_s[0] // 2, self.input_s[1] // 2, self.input_s[2] * 4]
            else:
                self.z_dim *= 2
                self.input_s = [self.input_s[0] // 2, self.input_s[1] // 2, self.input_s[2] * 2]
                
            self.levels.append(DiscriminatorCell(self.h_dim, self.input_s))
            
            
    def call(self, head, debug):
        features = list()
        
        for level, layer in zip(self.levels, head):
            features.append(level(layer))
            
        return features
    

    
    