#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import math

from zhusuan.kernels.base import Kernel

__all__ = [
    'LinearKernel',
]


class LinearKernel(Kernel):
    def __init__(self, kernel_num, covariates_num, name, variance_scale, dtype=tf.float32):
        super(LinearKernel, self).__init__(kernel_num, dtype)
        self._covariates_num = covariates_num
        self._variance = variance_scale / math.log(2) * tf.nn.softplus(
            tf.get_variable('StationaryKernel/variance_' + name, [kernel_num], self._dtype,
                            initializer=tf.zeros_initializer()))

    @property
    def variance(self):
        return self._variance

    def __call__(self, x, y):
        assert_ops = [
            tf.assert_greater_equal(
                tf.rank(x), 2,
                message='LinearKernel: rank(x) should be static and >=2'),
            tf.assert_greater_equal(
                tf.rank(y), 2,
                message='LinearKernel: rank(y) should be static and >=2'),
            tf.assert_equal(
                tf.rank(x), tf.rank(y),
                message='LinearKernel: x and y should have the same rank')]
        with tf.control_dependencies(assert_ops):
            rank = x.shape.ndims
            x = tf.expand_dims(tf.expand_dims(x, rank - 1), rank - 2)
            y = tf.expand_dims(tf.expand_dims(y, rank - 2), rank - 2)
        return self._variance * (x * y)

    def Kdiag(self, x):
        return self._variance * tf.square(x)
