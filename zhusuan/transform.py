#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

__all__ = ['normalizing_planar_flow']


# utils
def random_value(shape, mean=0, sd=0.05):
    return tf.random_normal(shape=shape, mean=mean, stddev=sd, dtype=tf.float32)


def semi_broadcast(x, base):
    base_shape = base.get_shape()
    base_ndim = base_shape.ndims
    x_shape = x.get_shape()
    x_ndim = int(x_shape.ndims)
    tx_shape = tf.concat([tf.shape(base)[:-x_ndim], tf.constant([1, 1], dtype=tf.int32)], 0)

    while x.get_shape().ndims < base_ndim:
        x = tf.expand_dims(x, 0)
    tx = tf.tile(x, multiples=tx_shape, name='semi_broadcast')
    return tx
    

def normalizing_planar_flow(sample, log_prob, iters):
    '''
        Perform Normalizing Planar Flow for the last dimension of input
            f(z) = z + h(z * w + b) * u
        with activation function tanh as well as the invertibility trick
        in (Danilo 2016)

        some shape example:
            z: [S, N, M, D]
            w: [D, 1]
            u: [1, D]
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
        log_det_ja.append(scalar * (tf.constant(1.0, dtype=tf.float32) - reduce_act * reduce_act) + tf.constant(1.0, dtype=tf.float32))
        z = z + tf.matmul(activation, para_u, name='update_calc')

    return (z, log_prob - sum(log_det_ja))