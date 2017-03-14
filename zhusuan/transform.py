#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

__all__ = [
    'linear_ar',
    'planar_nf',
    'iaf'
]


# utils
def random_value(shape, mean=0, sd=0.05):
    '''
    Return a random tensor
    '''
    return tf.random_normal(shape=shape, mean=mean, stddev=sd, dtype=tf.float32)


def semi_broadcast(x, base):
    '''
    shape(base) =  [i, ..., k, p, ..., q]
    shape(x)    =  [         , w, ..., z]
    return semi_broadcast of x
    shape(tx)   =  [i, ..., k, w, ..., z]
    '''
    base_shape = base.get_shape()
    base_ndim = base_shape.ndims
    x_shape = x.get_shape()
    x_ndim = int(x_shape.ndims)
    tx_shape = tf.concat([tf.shape(base)[:-x_ndim], tf.constant([1] * x_ndim, dtype=tf.int32)], 0)

    while x.get_shape().ndims < base_ndim:
        x = tf.expand_dims(x, 0)
    tx = tf.tile(x, multiples=tx_shape, name='semi_broadcast')
    return tx


# autoregressive neural network
def linear_ar(name, id, z, hidden=None):
    '''
    Implement the linear autoregressive network for inverse autoregressive 
    flow from (Kingma 2016).
    :name: A string, to define the 
    :id: Int
    :z: A N-D Tensor, the original sample
    :hidden: Linear autoregressive flow don't need hidden layer.
    '''
    # reshape z
    dynamic_z_shape = tf.shape(z)
    static_z_shape = z.get_shape()
    ndim = static_z_shape.ndims
    D = int(static_z_shape[ndim - 1])
    z_in = tf.reshape(z, [-1, D])

    # calculate diagonal mask
    mask = []
    for i in range(D):
        maski = [0] * D
        for j in range(D):
            maski[j] = int(i < j) * 1.0
        mask.append(maski)
    tfmask = tf.constant(mask, dtype=tf.float32)

    with tf.name_scope(name + '%d' % id):
        mW = tf.Variable(random_value([D, D]), name='mW')
        sW = tf.Variable(random_value([D, D]), name='mW')

        mW = tfmask * mW
        sW = tfmask * sW

        '''mW = semi_broadcast(mW, z)
        sW = semi_broadcast(sW, z)'''

        m = tf.matmul(z_in, mW)
        s = tf.matmul(z_in, sW)

        m = tf.reshape(m, dynamic_z_shape)
        s = tf.reshape(s, dynamic_z_shape)
        s = tf.exp(s)

    return (m, s)


def MADE(name, id, z, hidden):
    return None


def planar_nf(sample, log_prob, iters):
    '''
    Perform Normalizing Planar Flow for the last dimension of input
        f(z_t) = z_{t-1} + h(z_{t-1} * w_t + b_t) * u_t
    with activation function tanh as well as the invertibility trick
    in (Danilo 2016)
    :para sample: A N-D Tensor (N>=2) of shape (..., D), and normalizing 
    planar flow will perform flow on the last dimension.
    :para log_prob: A (N-1)-D Tensor of shape (...), which means 
    shape(log_prob)=shape(sample)[:-1].
    :para iters: A Int, which represents the number of successive flows
    (in the experiments of paper, K)

    :return: A N-D Tensor, the transformed sample of the origin input
    tensor
    :return: A (N-1)-D Tensor, the joint-prob of the transformed sample 

    Example:
    [input]
        sample(z0): [S, N, M, D]
        logpdf(q0): [S, N, M]
    [weight]
         weight(w): [D, 1]
         weight(u): [1, D]
    [output]
        sample(zk): [S, N, M, D]
        logpdf(qk): [S, N, M]
    '''
    input_x = sample

    # define parameters
    x_shape = input_x.get_shape()
    ndim = x_shape.ndims
    D = x_shape[ndim - 1]
    D = int(D)
    with tf.name_scope('flow_parameters'):
        para_bs = []
        para_us = []
        para_ws = []
        for iter in range(iters):
            para_b = tf.Variable(random_value([1]), name='para_b_%d' % iter)
            aux_u = tf.Variable(random_value([D, 1]), name='aux_u_%d' % iter)
            para_w = tf.Variable(random_value([D, 1]), name='para_w_%d' % iter)
            dot_prod = tf.matmul(para_w, aux_u, transpose_a=True)
            para_u = dot_prod + para_w / tf.matmul(para_w, para_w, transpose_a=True) \
                        * (tf.log(tf.exp(dot_prod) + 1) - 1 - dot_prod)
            para_u = tf.transpose(para_u, name='para_u_%d' % iter)
            para_bs.append(para_b)
            para_ws.append(para_w)
            para_us.append(para_u)

    # forward and log_det_jacobian
    z = input_x
    log_det_ja = []
    for iter in range(iters):
        scalar = tf.matmul(para_us[iter], para_ws[iter], name='scalar_calc')
        para_w = semi_broadcast(para_ws[iter], z)
        activation = tf.tanh(tf.matmul(z, para_w, name='score_calc') + para_bs[iter], name='activation_calc')
        para_u = semi_broadcast(para_us[iter], activation)

        reduce_act = tf.reduce_sum(activation, axis=-1)
        log_det_ja.append(tf.log(scalar * (tf.constant(1.0, dtype=tf.float32) - reduce_act * reduce_act) + tf.constant(1.0, dtype=tf.float32)))
        z = z + tf.matmul(activation, para_u, name='update_calc')

    return (z, log_prob - sum(log_det_ja))


def iaf(sample, hidden, log_prob, autoregressiveNN, iters, update='normal'):
    joint_prob = log_prob
    z = sample
    for iter in range(iters):
        m, s = autoregressiveNN('iaf', iter, sample, hidden)

        if update == 'gru':
            sigma = tf.sigmoid(s)
            z = sigma * z + (1 - sigma) * m
            joint_prob = joint_prob - tf.reduce_sum(tf.log(sigma), axis=-1)

        if update == 'normal':
            z = s * z + m
            joint_prob = joint_prob - tf.reduce_sum(tf.log(s), axis=-1)

    return (z, joint_prob)


