#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import math

from zhusuan.kernels.base import Kernel

__all__ = [
    'StationaryKernel',
    'RBFKernel',
    'ExponentialKernel'
]


class StationaryKernel(Kernel):
    def __init__(self, kernel_num, covariates_num, name, variance=1.0, length_scale=1.0, dtype=tf.float32):
        super(StationaryKernel, self).__init__(kernel_num, dtype)
        self._covariates_num = covariates_num
        self._variance = variance / math.log(2) * tf.nn.softplus(
            tf.get_variable('StationaryKernel/variance_' + name, [kernel_num], self._dtype,
                            initializer=tf.zeros_initializer()))
        self._length = length_scale / math.log(2) * tf.nn.softplus(
            tf.get_variable('StaticKernel/length_' + name, [kernel_num, covariates_num], self._dtype,
                            initializer=tf.zeros_initializer()))

    def __call__(self, x, y):
        raise NotImplementedError()

    @property
    def covariates_num(self):
        return self._covariates_num

    @property
    def variance(self):
        return self._variance

    @property
    def length_scale(self):
        return self._length

    def _scaled_square_distance(self, x, y):
        assert_ops = [
            tf.assert_greater_equal(
                tf.rank(x), 2,
                message='StationaryKernel: rank(x) should be static and >=2'),
            tf.assert_greater_equal(
                tf.rank(y), 2,
                message='StationaryKernel: rank(y) should be static and >=2'),
            tf.assert_equal(
                tf.rank(x), tf.rank(y),
                message='RBFKernel: x and y should have the same rank')]
        with tf.control_dependencies(assert_ops):
            rank = x.shape.ndims
            x = tf.expand_dims(tf.expand_dims(x, rank - 1), rank - 2)
            y = tf.expand_dims(tf.expand_dims(y, rank - 2), rank - 2)
            length_scale = tf.reshape(self._length,
                                      [1] * (rank - 2) + [self._kernel_num, 1, 1, self._covariates_num])
            dist = tf.reduce_sum(tf.square(x - y) / length_scale, axis=-1)
        return dist

    def Kdiag(self, x):
        rank = x.shape.ndims
        return tf.reshape(self._variance, [1, self.kernel_num, 1]) * tf.tile(
            tf.expand_dims(tf.ones_like(tf.reduce_sum(x, -1)), -2), [1] * (rank - 2) + [self.kernel_num, 1])


class RBFKernel(StationaryKernel):
    def __init__(self, kernel_num, covariates_num, name, variance=1.0, length_scale=1.0, dtype=tf.float32):
        super(RBFKernel, self).__init__(kernel_num, covariates_num, name, variance, length_scale, dtype)

    def __call__(self, x, y):
        rank = x.shape.ndims
        return tf.reshape(self._variance, [1] * (rank - 3) + [-1, 1, 1]) * \
            tf.exp(-0.5 * self._scaled_square_distance(x, y))


class ExponentialKernel(StationaryKernel):
    def __init__(self, kernel_num, covariates_num, name, variance=1.0, length_scale=1.0, dtype=tf.float32):
        super(ExponentialKernel, self).__init__(kernel_num, covariates_num, name, variance, length_scale, dtype)

    def __call__(self, x, y):
        rank = x.shape.ndims
        return tf.reshape(self._variance, [1] * (rank - 3) + [-1, 1, 1]) * \
            tf.exp(-0.5 * tf.sqrt(self._scaled_square_distance(x, y) + 1e-8))
