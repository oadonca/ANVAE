# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:30:18 2020

@author: Octavian
"""
import tensorflow as tf
import modules
import numpy as np

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
                tf.keras.layers.DepthwiseConv2D(kernel_size=5, padding='same', use_bias=False),
                # tf.keras.layers.Conv2D(expanded_z_dim, kernel_size=5, padding="same", groups=expanded_z_dim, use_bias=False),
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
                tf.keras.layers.Conv2DTranspose(z_dim, kernel_size=3, strides=scale, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('swish')
            ]
        )
        
    def call(self, x):
        return self.seq(x)
        
    
def AbsoluteVariationalBlock(feature_shape, latent_channels):
    channels = feature_shape[-1]
    return modules.AbsoluteVariationalBlock(sample = modules.AbsoluteVariational(
        parameters = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(channels, kernel_size=1),
                tf.keras.layers.Activation('swish'),
                tf.keras.layers.Conv2D(latent_channels*2, kernel_size=1)
            ]
        )), 
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
        ))

def RelativeVariationalBlock(previous_shape, feature_shape, latent_channels):
    channels = feature_shape[-1]
    return modules.RelativeVariationalBlock(
        sample = modules.RelativeVariational(
            absolute_parameters = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(channels, kernel_size=1),
                    tf.keras.layers.Activation('swish'),
                    tf.keras.layers.Conv2D(latent_channels*2, kernel_size=1)
                ]    
            ),
            relative_parameters = tf.keras.Sequential(
                [
                    # x[0] = previous, x[1] = previous
                    # tf.keras.layers.Lambda(lambda previous, feature: (tf.concat([previous,feature], dim = -1))),
                    tf.keras.layers.Conv2D(channels, kernel_size=1),
                    tf.keras.layers.Activation('swish'),
                    tf.keras.layers.Conv2D(latent_channels*2, kernel_size=1)                    
                ]
            )
        ), 
        decoded_sample = tf.keras.Sequential(
            [
                modules.RandomFourier(8),
                tf.keras.layers.Conv2D(channels, kernel_size=1, use_bias=False),
                DecoderCell(channels)
            ]
        ), 
        computed = tf.keras.Sequential(
            [
                # x[0] = decoded_sample, x[1] = previous
                # tf.keras.layers.Lambda(lambda decoded_sample, previous: (tf.concat([decoded_sample, previous], axis=1))),
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
        
        self.latent_height = example_features[-1].shape[-3]
        self.latent_width = example_features[-1].shape[-2]
        
        relative_variational_blocks = []
        upsampled_blocks = []
        for level_index, (level_size, example_feature) in enumerate(zip(level_sizes, reversed(example_features))):
            inner_blocks = []
            for block_index in range(1 if level_index == 0 else 0, level_size):
                relative_variational_block = RelativeVariationalBlock(
                    previous.shape,
                    example_feature.shape,
                    latent_channels
                )
                previous, _ = relative_variational_block(previous, example_feature)
                inner_blocks.append(relative_variational_block)
                
            relative_variational_blocks.append(tf.keras.Sequential(inner_blocks))
            upsample = UpsampleBlock(previous.shape[-1], 8 if level_index == len(level_sizes)-1 else 2)
            previous = upsample(previous)
            upsampled_blocks.append(upsample)
            
        self.relative_variational_blocks = tf.keras.Sequential(relative_variational_blocks)
        self.upsampled_blocks = tf.keras.Sequential(upsampled_blocks)
        
        self.n_mixture_components = 5
        
        self.image = tf.keras.Sequential(
            [
                DecoderCell(previous.shape[-1]),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(3*3*self.n_mixture_components, kernel_size=1),
                tf.keras.layers.Lambda(
                    lambda x: np.reshape(x, (-1, 3, 3 * self.n_mixture_components, 32, 32))
                ),
                tf.keras.layers.Lambda(lambda x: np.transpose(x, (0, 3, 4, 1, 2))),
                tf.keras.layers.Lambda(lambda x: np.split(x, 3, axis=-1)),
                tf.keras.layers.Lambda(lambda logits, unlimited_loc, unlimited_scale: (
                    logits,
                    tf.math.tanh(unlimited_loc),
                    tf.math.softplus(unlimited_scale)
                ))
            ]
        )
        
    def call(self, features):
        head, kl = self.absolute_variational_block(features[-1])
        
        kl_losses = [kl]
        
        for feature, blocks, upsampled in zip(reversed(features), self.relative_variational_blocks, self.upsampled_blocks):
            for block in blocks:
                head, relative_kl = block(head, feature)
                kl.losses.append(relative_kl)
            head = upsampled(head)
            
        return (
            self.image(head),
            kl_losses
        )
            
        