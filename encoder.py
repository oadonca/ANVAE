# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:30:12 2020

@author: Octavian
"""

import tensorflow as tf
import modules

class EncoderCell(tf.keras.Model):
    def __init__(self, input_shape, z_dim):
        super(EncoderCell, self).__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('swish'),
                tf.keras.layers.Conv2D(z_dim, kernel_size=3, strides=1, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('swish'),
                tf.keras.layers.Conv2D(z_dim, kernel_size=3, strides=1, padding="same"),
                modules.SEKeras(z_dim)
            ]
        )
        
    def call(self, x):
        return x + self.seq(x)

class Encoder(tf.keras.Model):
    def __init__(self, latent_spaces, input_shape):
        super(Encoder, self).__init__()
        self.z_dim = input_shape[2]
        self.input_s = input_shape
        self.levels = tf.keras.Sequential()
        for level in range(latent_spaces):
            if(level == 0):
                self.z_dim *= 8
                self.input_s = [self.input_s[0] // 2, self.input_s[1] // 2, self.input_s[2] * 8]
                self.levels.add(tf.keras.layers.InputLayer(self.input_s))
                self.levels.add(tf.keras.layers.Conv2D(filters=self.z_dim, kernel_size=1, strides=2, use_bias=False))
            else:
                self.z_dim *= 2
                self.input_s = [self.input_s[0] // 2, self.input_s[1] // 2, self.input_s[2] * 2]
                self.levels.add(tf.keras.layers.InputLayer(self.input_s))
                self.levels.add(tf.keras.layers.Conv2D(filters=self.z_dim, kernel_size=1, strides=2, use_bias=False))
                
            print(self.input_s)
            self.levels.add(EncoderCell(tf.TensorShape(self.input_s), self.z_dim))
            
    def call(self, head):
        features = list()
        for level in self.levels:
            head = level(head)
            features.append(head)
        return features
    
