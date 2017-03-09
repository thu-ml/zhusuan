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
                 group_event_ndims,
                 is_reparameterized=True,
                 check_numerics=False):
        self._sample_output = forward(input)
        self._log_prob_output = -log_det_jacobian(input)
        self._check_numerics = check_numerics
        super(Transformed, self).__init__(
            dtype=tf.float32,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            group_event_ndims=group_event_ndims)

    def _value_shape(self):
        raise NotImplementedError()

    def _get_value_shape(self):
        raise NotImplementedError()

    def _batch_shape(self):
        raise NotImplementedError()

    def _get_batch_shape(self):
        raise NotImplementedError()

    def _sample(self, n_samples):
        return self._sample_output

    def _log_prob(self, given):
        return self._log_prob_output

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class NormalizingPlanarFlow(Transformed):
    '''
        Perform Normalizing Flow for the last dimension of input

    '''
    def __init__(self,
                 input,
                 iters,
                 actiavtion=tf.tanh):

        @zs.reuse('model')
        def forward(input_x):
            z = input_x
            x_shape = input_x.get_shape()
            ndim = x_shape.ndims
            D = x_shape[ndim - 1]

            w = tf.Variable()
            for iter in range(iters):
                z = z

            return z

        def