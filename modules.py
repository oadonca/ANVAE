# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:34:58 2020

@author: Octavian
"""

import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import functools


def batchNorm(layer_dict, name="batch_norm"):
    batch_mean, batch_var = tf.compat.v1.nn.moments(x=layer_dict['cur_input'], axes=[0, 1, 2])
    layer_dict['cur_input'] = tf.compat.v1.nn.batch_normalization(x=layer_dict['cur_input'], mean=batch_mean, variance=batch_var, offset=None, scale=None, variance_epsilon=1e-3)
    return layer_dict['cur_input']

class SEKeras(tf.keras.layers.Layer):
    def __init__(self, z_dim, red = 16):
        super(SEKeras, self).__init__()
        if (z_dim == 0):
            reduced_channels = 1
        else:
            reduced_channels = max(z_dim // red, int(z_dim ** 0.5))
        self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.AveragePooling2D(1),
                    tf.keras.layers.Dense(reduced_channels, use_bias=False),
                    tf.keras.layers.ReLU(),
                    tf.keras.layers.Dense(z_dim, use_bias=False),
                    tf.keras.layers.Activation('sigmoid')
                ]
            )
        
    def call(self, x):
        return x * self.fc(x)

def swish(layer_dict, name="swish"):
    layer_dict['cur_input'] = tf.compat.v1.nn.swish(layer_dict['cur_input'])
    return layer_dict['cur_input']

def Global_Average_Pooling(x):
    return tf.keras.layers.GlobalAveragePooling2D()(x)

def Fully_connected(x, units, name='fully_connected') :
    with tf.name_scope(name) :
        return tf.compat.v1.layers.dense(inputs=x, use_bias=False, units=units)
    
def Relu(x):
    return tf.compat.v1.nn.relu(x)

def Sigmoid(x) :
    return tf.compat.v1.nn.sigmoid(x)

def SE(layer_dict, out_dim, ratio = 16, name="SE"):
    with tf.name_scope(name) :
        squeeze = Global_Average_Pooling(layer_dict['cur_input'])

        dim = out_dim
        if out_dim / ratio < 1:
            dim = 1            

        excitation = Fully_connected(squeeze, units=dim / ratio, name=name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, name=name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        layer_dict['cur_input'] = layer_dict['cur_input'] * excitation

        return layer_dict['cur_input']
            
def get_shape4D(in_val):
    """
    Return a 4D shape
    Args:
        in_val (int or list with length 2)
    Returns:
        list with length 4
    """
    # if isinstance(in_val, int):
    return [1] + get_shape2D(in_val) + [1]

def get_shape2D(in_val):
    """
    Return a 2D shape 
    Args:
        in_val (int or list with length 2) 
    Returns:
        list with length 2
    """
    in_val = int(in_val)
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))
    
def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

def softplus(inputs, name):
    return tf.compat.v1.log(1 + tf.exp(inputs), name=name)

def encoder_cell(inputs, weights, biases, is_training, dim, wd=0, bn=False, name='encoder_cell', init_w=tf.keras.initializers.he_normal(), change_dim = False):
        # init_w = tf.keras.initializers.he_normal()
        layer_dict = {}
        layer_dict['cur_input'] = inputs
        with tf.compat.v1.variable_scope(name):
            
            if (change_dim):
                conv(layer_dict, weights, biases, strides = 2, filter_size=3)
                
            batchNorm(layer_dict)
            
            swish(layer_dict)
                
            conv(layer_dict, weights, biases, out_c = dim, strides = 1, filter_size = 3, padding="SAME")
            
            batchNorm(layer_dict)
            
            swish(layer_dict)
            
            conv(layer_dict, weights, biases, out_c = dim, strides = 1, filter_size = 3, padding="SAME")
            
            SE(layer_dict, dim)
            
            return layer_dict['cur_input']
    
    
def conv2_wrapper(x, W, b, nl=tf.identity, strides=1, padding="SAME"):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return nl(x)

def conv(layer_dict, weights, biases, strides, filter_size, channel_mult = 1, padding="VALID", out_c=None, trainable=True, name='conv', init_w=tf.keras.initializers.he_normal(), init_b=tf.zeros_initializer()):
    inputs = layer_dict['cur_input']
    
    if (out_c == None):
        out_c=inputs[3]
    
    output = conv2_wrapper(inputs, weights, biases, strides=strides, padding=padding)
    
    layer_dict['cur_input'] = output
    layer_dict[name] = layer_dict['cur_input']
    return layer_dict['cur_input']

def linear(inputs, weights, biases, nl=tf.identity, name='linear'):
    
    inputs = inputs
    
    inputs = batch_flatten(inputs)
    
    output = tf.add(tf.matmul(inputs, weights), biases)
    
    output = nl(output)
    
    return output
    
def tf_sample_standard_diag_guassian(b_size, n_code):
    mean_list = [0.0 for i in range(0, n_code)]
    std_list = [1.0 for i in range(0, n_code)]
    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=mean_list,
        scale_diag=std_list)
    samples = mvn.sample(sample_shape=(b_size,), seed=None, name='sample')
    return samples

def tf_sample_diag_guassian(mean, std, b_size, n_code):
    mean_list = [0.0 for i in range(0, n_code)]
    std_list = [1.0 for i in range(0, n_code)]
    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=mean_list,
        scale_diag=std_list)
    samples = mvn.sample(sample_shape=(b_size,), seed=None, name='sample')
    samples = mean +  tf.multiply(std, samples)

    return samples

class AbsoluteVariationalBlock(tf.keras.Model):
    def __init__(self, sample, decoded_sample, computed):
        super(AbsoluteVariationalBlock, self).__init__()
        self.sample = sample
        self.decoded_sample = decoded_sample
        self.computed = computed

    def call(self, head):
        sample, kl = self.sample(head)
        computed = self.computed(
            self.decoded_sample(sample)
        )
        return computed, kl

    def generated(self, shape, prior_std):
        return self.computed(
            self.decoded_sample(
                self.sample.generated(shape, prior_std)
            )
        )

class RelativeVariationalBlock(tf.keras.Model):
    def __init__(self, sample, decoded_sample, computed):
        super(RelativeVariationalBlock, self).__init__()
        self.sample = sample
        self.decoded_sample = decoded_sample
        self.computed = computed

    def call(self, previous, feature):
        sample, kl = self.sample(previous, feature)
        ccat = tf.concat([self.decoded_sample(sample), previous], axis=-1)
        computed = self.computed(ccat)
        return computed, kl

    def generated(self, previous, prior_std):
        return self.computed(
            self.decoded_sample(
                self.sample.generated(previous, prior_std)
            ),
            previous,
        )


class AbsoluteVariational(tf.keras.Model):
    def __init__(self, parameters):
        super().__init__()
        self.variational_parameters = parameters

    @staticmethod
    def sample(mean, log_variance):
        std = tf.math.exp(0.5 * log_variance)
        return mean + tf.random.normal(shape=std.shape) * std

    @staticmethod
    def kl(mean, log_variance):
        loss = -0.5 * (1 + log_variance - mean ** 2 - tf.math.exp(log_variance))
        # return loss.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        return tf.keras.backend.mean(loss)

    def call(self, feature):
        temp = self.variational_parameters(feature)
        mean, log_variance = tf.split(temp, 2, -1)
        # print('absolute mean:', mean.view(-1)[:5])
        # print('absolute log_variance:', log_variance.view(-1)[:5])
        return (
            AbsoluteVariational.sample(mean, log_variance),
            AbsoluteVariational.kl(mean, log_variance),
        )

    def generated(self, shape, prior_std):
        return tf.random.normal(shape) * prior_std
    
class RelativeVariational(tf.keras.Model):
    def __init__(self, absolute_parameters, relative_parameters):
        super(RelativeVariational, self).__init__()
        self.absolute_parameters = absolute_parameters
        self.relative_parameters = relative_parameters

    @staticmethod
    def kl(mean, log_variance, delta_mean, delta_log_variance):
        var = tf.math.exp(log_variance)
        delta_var = tf.math.exp(delta_log_variance)
        loss = -0.5 * (
            1 + delta_log_variance - delta_mean ** 2 / var - delta_var
        )
        # return loss.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        return tf.keras.backend.mean(loss)

    def call(self, previous, feature):
        temp = self.absolute_parameters(previous)
        mean, log_variance = tf.split(temp, 2, -1)
        ccat = tf.concat([previous,feature], axis = -1)
        temp = self.relative_parameters(ccat)
        delta_mean, delta_log_variance = tf.split(temp, 2, -1)
        # print('relative mean:', (mean + delta_mean).view(-1)[:5])
        # print('relative log_variance:', (log_variance + delta_log_variance).view(-1)[:5])
        return (
            AbsoluteVariational.sample(
                mean + delta_mean, log_variance + delta_log_variance
            ),
            RelativeVariational.kl(
                mean, log_variance, delta_mean, delta_log_variance
            ),
        )

    def generated(self, previous, prior_std):
        temp = self.absolute_parameters(previous)
        mean, log_variance = tf.split(temp, 2, -1)
        return AbsoluteVariational.sample(
            mean, log_variance + 2 * np.log(prior_std)
        )

class RandomFourier(tf.keras.layers.Layer):
    def __init__(self, fourier_channels):
        super(RandomFourier, self).__init__()
        if fourier_channels % 2 != 0:
            raise ValueError('Out channel must be divisible by 4')

        self.random_matrix = tf.random.normal((2, fourier_channels // 2))
        

    @staticmethod
    def gridspace(x):
        h = x.shape[1]
        w = x.shape[2]
        grid_y, grid_x = tf.meshgrid(
            tf.linspace(0, 1, num=h),
            tf.linspace(0, 1, num=w)
        )
        return (
            tf.cast(tf.tile(tf.expand_dims(tf.stack([grid_y, grid_x]), 0), (x.shape[0], 1, 1, 1)), x.dtype)
        )

    def call(self, x):
        gridspace = RandomFourier.gridspace(x)
        projection = (
            (2 * np.pi * tf.transpose(gridspace, perm=[0, 3, 2, 1])) @ self.random_matrix
        )
        return tf.concat([
            x,
            tf.math.sin(projection),
            tf.math.cos(projection),
        ], axis=-1)


"""
def conv(filter_size,
         out_dim,
         layer_dict,
         inputs=None,
         pretrained_dict=None,
         stride=1,
         dilations=[1, 1, 1, 1],
         bn=False,
         nl=tf.identity,
         init_w=None,
         init_b=tf.zeros_initializer(),
         use_bias=True,
         padding='SAME',
         pad_type='ZERO',
         trainable=True,
         is_training=None,
         wd=0,
         name='conv',
         add_summary=False):
    if inputs is None:
        inputs = layer_dict['cur_input']
    stride = get_shape4D(stride)
    in_dim = inputs.get_shape().as_list()[-1]
    
    # (3, 3, 1, out_dim)
    filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]

    if padding == 'SAME' and pad_type == 'REFLECT':
        pad_size_1 = int((filter_shape[0] - 1) / 2)
        pad_size_2 = int((filter_shape[1] - 1) / 2)
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_size_1, pad_size_1], [pad_size_2, pad_size_2], [0, 0]],
            "REFLECT")
        padding = 'VALID'

    with tf.compat.v1.variable_scope(name):
        if pretrained_dict is not None and name in pretrained_dict:
            try:
                load_w = pretrained_dict[name][0]
            except KeyError:
                load_w = pretrained_dict[name]['weights']
            print('Load {} weights!'.format(name))

            load_w = np.reshape(load_w, filter_shape)
            init_w = tf.constant_initializer(load_w)

        weights = tf.compat.v1.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w, 
                                  trainable=trainable,
                                  regularizer=None)
        
        print("\nCONV WEIGHT: {}\n".format(weights.shape))
        
        if add_summary:
            tf.summary.histogram(
                'weights/{}'.format(name), weights, collections = ['train'])

        outputs = tf.compat.v1.nn.conv2d(inputs,
                               filter=weights,
                               strides=stride,
                               padding=padding,
                               use_cudnn_on_gpu=True,
                               data_format="NHWC",
                               dilations=dilations,
                               name='conv2d')

        if use_bias:
            if pretrained_dict is not None and name in pretrained_dict:
                try:
                    load_b = pretrained_dict[name][1]
                except KeyError:
                    load_b = pretrained_dict[name]['biases']
                print('Load {} biases!'.format(name))

                load_b = np.reshape(load_b, [out_dim])
                init_b = tf.constant_initializer(load_b)

            biases = tf.compat.v1.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)
            outputs += biases

        # if bn is True:
        #     outputs = layers.batch_norm(outputs, train=is_training, name='bn')

        layer_dict['cur_input'] = nl(outputs)
        layer_dict[name] = layer_dict['cur_input']
        return layer_dict['cur_input']
"""
"""
def linear(out_dim,
           layer_dict=None,
           inputs=None,
           init_w=None,
           init_b=tf.zeros_initializer(),
           wd=0,
           name='Linear',
           nl=tf.identity):
    with tf.compat.v1.variable_scope(name):
        if inputs is None:
            assert layer_dict is not None
            inputs = layer_dict['cur_input']
        inputs = batch_flatten(inputs)
        in_dim = inputs.get_shape().as_list()[1]
        
        if wd > 0:
            regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
        else:
            regularizer=None
        
        weights = tf.compat.v1.get_variable('weights',
                                  shape=[in_dim, out_dim],
                                  # dtype=None,
                                  initializer=init_w,
                                  regularizer=regularizer,
                                  trainable=True)
        biases = tf.compat.v1.get_variable('biases',
                                  shape=[out_dim],
                                  # dtype=None,
                                  initializer=init_b,
                                  regularizer=None,
                                  trainable=True)
        # print('init: {}'.format(weights))
        act = tf.compat.v1.nn.xw_plus_b(inputs, weights, biases)
        result = nl(act, name='output')
        if layer_dict is not None:
            layer_dict['cur_input'] = result
            
        return result
"""