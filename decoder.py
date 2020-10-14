# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:30:18 2020

@author: Octavian
"""
import tensorflow as tf
import modules

class DecoderCell(tf.keras.Model):
    def __init__(self, z_dim):
        super(DecoderCell, self).__init__()
        expanded_z_dim = z_dim * 6
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(expanded_z_dim, kernel_size=1, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('swish'),
                tf.keras.layers.Conv2D(expanded_z_dim, kernel_size=5, groups=expanded_z_dim, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('swish'),
                tf.keras.layers.Conv2D(z_dim, kernel_size=1, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                modules.SEKeras(z_dim)
            ]
            
        )
        
    def call(self, x):
        return x + self.seq(x)
    
class UpsampleBlock(tf.keras.Model):
    def __init__(self, z_dim, scale):
        super(UpsampleBlock, self).__init__()
        
        if scale is None:
            scale = 2
        
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(z_dim, kernel_size=3, stride=scale, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('swish')
            ]
        )
        
    
def AbsoluteVariationalBlock(feature_shape, latent_channels):
    channels = feature_shape[-1]
    return modules.AbsoluteVariationalBlock(sample = modules.AbsoluteVariational(
        parameters = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(channels, kernel_size=1),
                tf.keras.layers.Activation('swish'),
                tf.keras.layers.Conv2D(latent_channels*2, kernel_size=1)
            ]
        ), 
        decoded_sample = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(channels, kernel_size=1, use_bias=False),
                DecoderCell(channels)
            ]
        ), 
        computed = tf.keras.Sequential(
            [
                DecoderCell(channels)
            ]
        )))

def RelativeVariationalBlock(previous_shape, feature_shape, latent_channels):
    channels = feature_shape[-1]
    return modules.RelativeVariationalBlock(
        sample = modules.RelativeVariational(
            absolute_parameters = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(channels, kernel_size=1),
                tf.keras.layers.Activation('swish'),
                tf.keras.layers.Conv2D()
            ]    
        )), 
        decoded_sample = tf.keras.Sequential(
            [
                modules.RandomFourier(8),
                tf.keras.layers.Conv2D(channels, kernel_size=1, use_bias=False),
                DecoderCell(channels)
            ]
        ), 
        computed = tf.keras.Sequential(
            [
                lambda decoded_sample, previous: (tf.concat([decoded_sample, previous], axis=1))   ,
                DecoderCell(channels+previous_shape[1]),
                tf.keras.layers.Conv2D(channels, kernel_size=1)
            ]
        )  
    )
    
class Decoder(tf.keras.Model):
    def __init__(self, example_features, latent_channels, level_sizes):
        super(Decoder, self).__init__()
        self.absolute_variational_block = AbsoluteVariationalBlock(example_features[-1].shape, latent_channels)
        previous, _ = self.absolute_variational_block(example_features[-1])