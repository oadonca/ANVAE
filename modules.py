# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:34:58 2020

@author: Octavian
"""

import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def batchNorm(layer_dict, name="batch_norm"):
    batch_mean, batch_var = tf.nn.moments(x=layer_dict['cur_input'])
    layer_dict['cur_input'] = tf.nn.batch_normalization(x=layer_dict['cur_input'], mean=batch_mean, variance=batch_var, offset=None, scale=None, variance_epsilon=1e-3)
    return layer_dict['cur_input']

def swish(layer_dict, name="swish"):
    layer_dict['cur_input'] = tf.nn.swish(layer_dict['cur_input'])
    return layer_dict['cur_input']

def Global_Average_Pooling(x):
    return tf.keras.layers.GlobalAveragePooling2D(x)

def Fully_connected(x, units, name='fully_connected') :
    with tf.name_scope(name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)
    
def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def SE(layer_dict, out_dim, ratio = 16, name="SE"):
    with tf.name_scope(name) :
        squeeze = Global_Average_Pooling(layer_dict['cur_input'])

        excitation = Fully_connected(squeeze, units=out_dim / ratio, name=name+'_fully_connected1')
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
    return tf.log(1 + tf.exp(inputs), name=name)

def encoder_cell(inputs, is_training, dim, wd=0, bn=False, name='encoder_cell', init_w=tf.keras.initializers.he_normal()):
        # init_w = tf.keras.initializers.he_normal()
        layer_dict = {}
        layer_dict['cur_input'] = inputs
        with tf.variable_scope(name):
            batchNorm(layer_dict)
            swish(layer_dict)
            conv(filter_size=3, out_dim=dim, name='conv1', add_summary=False, layer_dict=layer_dict, bn=bn, nl=tf.nn.relu, init_w=init_w, padding='SAME', pad_type='ZERO', is_training=is_training, wd=0)
            batchNorm(layer_dict)
            swish(layer_dict)
            conv(filter_size=3, out_dim=dim, name='conv2', add_summary=False, layer_dict=layer_dict, bn=bn, nl=tf.nn.relu, init_w=init_w, padding='SAME', pad_type='ZERO', is_training=is_training, wd=0)
            SE(layer_dict, dim)

            return layer_dict['cur_input']
    
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
    filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]

    if padding == 'SAME' and pad_type == 'REFLECT':
        pad_size_1 = int((filter_shape[0] - 1) / 2)
        pad_size_2 = int((filter_shape[1] - 1) / 2)
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_size_1, pad_size_1], [pad_size_2, pad_size_2], [0, 0]],
            "REFLECT")
        padding = 'VALID'

    with tf.variable_scope(name):
        if pretrained_dict is not None and name in pretrained_dict:
            try:
                load_w = pretrained_dict[name][0]
            except KeyError:
                load_w = pretrained_dict[name]['weights']
            print('Load {} weights!'.format(name))

            load_w = np.reshape(load_w, filter_shape)
            init_w = tf.constant_initializer(load_w)

        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable,
                                  regularizer=None)
        if add_summary:
            tf.summary.histogram(
                'weights/{}'.format(name), weights, collections = ['train'])

        outputs = tf.nn.conv2d(inputs,
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

            biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)
            outputs += biases

        # if bn is True:
        #     outputs = layers.batch_norm(outputs, train=is_training, name='bn')

        layer_dict['cur_input'] = nl(outputs)
        layer_dict[name] = layer_dict['cur_input']
        return layer_dict['cur_input']

def linear(out_dim,
           layer_dict=None,
           inputs=None,
           init_w=None,
           init_b=tf.zeros_initializer(),
           wd=0,
           name='Linear',
           nl=tf.identity):
    with tf.variable_scope(name):
        if inputs is None:
            assert layer_dict is not None
            inputs = layer_dict['cur_input']
        inputs = batch_flatten(inputs)
        in_dim = inputs.get_shape().as_list()[1]
        
        if wd > 0:
            regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
        else:
            regularizer=None
        weights = tf.get_variable('weights',
                                  shape=[in_dim, out_dim],
                                  # dtype=None,
                                  initializer=init_w,
                                  regularizer=regularizer,
                                  trainable=True)
        biases = tf.get_variable('biases',
                                  shape=[out_dim],
                                  # dtype=None,
                                  initializer=init_b,
                                  regularizer=None,
                                  trainable=True)
        # print('init: {}'.format(weights))
        act = tf.nn.xw_plus_b(inputs, weights, biases)
        result = nl(act, name='output')
        if layer_dict is not None:
            layer_dict['cur_input'] = result
            
        return result
    
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
    print('7'*80)
    print(tf.shape(samples))

    return samples