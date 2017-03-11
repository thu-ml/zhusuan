#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .base import *

__all__ = ['Transformed']


class Transformed(Distribution):
    def __init__(self,
                 input,
                 forward,
                 log_det_jacobian,
                 group_event_ndims=0,
                 value_ndims=1,
                 is_reparameterized=True,
                 check_numerics=False):
        self._input = input
        self._sample_output = forward(input)
        self._log_prob_output = -log_det_jacobian(input)

        if self._input.get_shape() != self._sample_output.get_shape():
            raise ValueError(
            "input and output should have the same shape"
            "({} vs. {})".format(
                self._input.get_shape(), self._sample_output.get_shape()))

        self._value_ndims = value_ndims
        self._check_numerics = check_numerics
        super(Transformed, self).__init__(
            dtype=tf.float32,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            group_event_ndims=group_event_ndims)

    def _value_shape(self):
        dynamic_shape = tf.shape(self._input)
        return dynamic_shape[-self._value_ndims:0]

    def _get_value_shape(self):
        static_shape = self._input.get_shape()
        return static_shape[-self._value_ndims:0]

    def _batch_shape(self):
        dynamic_shape = tf.shape(self._input)
        return dynamic_shape[:self._value_ndims]

    def _get_batch_shape(self):
        static_shape = tf.shape(self._input)
        return static_shape[:self._value_ndims]

    def _sample(self, n_samples):
        return self._sample_output

    def _log_prob(self, given):
        return self._log_prob_output

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


def NormalizingPlanarFlow(input, iters, group_event_ndims):
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
    x_shape = input.get_shape()
    ndim = x_shape.ndims
    D = x_shape[ndim - 1]
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

    def forward(input_x):
        z = input_x
        for iter in range(iters):
            z = z + tf.matmul(tf.tanh(tf.matmul(z, para_ws[iter]) + para_bs[iter]), para_us[iter])
        return z

    def log_det_jacobian(input_x):
        z = input_x
        log_det_ja = []
        for iter in range(iters):
            scalar = tf.matmul(para_us[iter], para_ws[iter])
            log_det_ja.append(scalar * (1 - tf.tanh(tf.matmul(z, para_ws[iter])) + para_bs[iter]) + 1.0)
            z = z + tf.matmul(tf.tanh(tf.matmul(z, para_ws[iter]) + para_bs[iter]), para_us[iter])
        return sum(log_det_ja)

    return Transformed(input, forward, log_det_jacobian, group_event_ndims)


def random_value(shape, mean=0, sd=0.05):
    return tf.random_normal(shape=shape, mean=mean, sdddev=sd, dtype=tf.float32)