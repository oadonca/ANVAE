# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:30:12 2020

@author: Octavian
"""

import tensorflow as tf
import tensorflow_probability as tfp
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
        self.levels = []
        for level in range(latent_spaces):
            
            curLevel = tf.keras.Sequential()
            
            if(level == 0):
                self.z_dim *= 4
                curLevel.add(tf.keras.layers.InputLayer(self.input_s))
                curLevel.add(tf.keras.layers.Conv2D(filters=self.z_dim, kernel_size=1, strides=2, use_bias=False))
                self.input_s = [self.input_s[0] // 2, self.input_s[1] // 2, self.input_s[2] * 4]
            else:
                self.z_dim *= 2
                curLevel.add(tf.keras.layers.InputLayer(self.input_s))
                curLevel.add(tf.keras.layers.Conv2D(filters=self.z_dim, kernel_size=1, strides=2, use_bias=False))
                self.input_s = [self.input_s[0] // 2, self.input_s[1] // 2, self.input_s[2] * 2]
                
            curLevel.add(tf.keras.layers.Dense(self.z_dim, kernel_regularizer=tf.keras.regularizers.L2(), activity_regularizer=tf.keras.regularizers.L1()))
            curLevel.add(EncoderCell(tf.TensorShape(self.input_s), self.z_dim))
            
            self.levels.append(curLevel)
            
    def call(self, head, debug):
        features = list()
        for level in self.levels:
            head = level(head)
            features.append(head)
            if debug:
                print("Head")
                print(head)
        return features
    

    
    