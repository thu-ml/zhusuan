#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import math

from zhusuan.kernels.base import Kernel

__all__ = [
    'StaticKernel',
    'WhiteKernel',
    'ConstantKernel',
]


class StaticKernel(Kernel):
    def __init__(self, kernel_num, name, variance=1.0, dtype=tf.float32):
        super(StaticKernel, self).__init__(kernel_num, dtype)
        self._variance = variance / math.log(2) * \
                         tf.get_variable('StaticKernel/variance_' + name, [kernel_num],
                                         self._dtype, initializer=tf.ones_initializer())

    def __call__(self, x, y):
        raise NotImplementedError()

    def Kdiag(self, x):
        shape = list(x.get_shape[:-2]) + [self._kernel_num, x.get_shape()[-2]]
        return self._variance * tf.ones(shape, dtype=tf.float32)


class WhiteKernel(StaticKernel):
    def __init__(self, kernel_num, name, dtype=tf.float32):
        super(WhiteKernel, self).__init__(kernel_num, name, dtype)

    def __call__(self, x, y):
        assert_ops = [tf.assert_greater_equal(tf.rank(x), 2, message='WhiteKernel: rank(x) should be static and >= 2'),
                      tf.assert_greater_equal(tf.rank(y), 2, message='WhiteKernel: rank(y) should be static and >= 2'),
                      tf.assert_equal(tf.shape(x)[:-2], tf.shape(y)[:-2],
                                      message='WhiteKernel: shapes of x and y should match')]
        with tf.control_dependencies(assert_ops):
            batch_shape = list(x.get_shape()[:-2]) + [self._kernel_num]
            return tf.cond(x == y,
                           tf.eye(x.get_shape()[-2], batch_shape=batch_shape, dtype=self._dtype),
                           tf.zeros(batch_shape + [x.get_shape()[-2], y.get_shape()[-2]], dtype=self._dtype))


class ConstantKernel(StaticKernel):
    def __init__(self, kernel_num, name, variance=1.0, dtype=tf.float32):
        super(ConstantKernel, self).__init__(kernel_num, name, variance, dtype)

    def __call__(self, x, y):
        assert_ops = [tf.assert_greater_equal(tf.rank(x), 2, message='WhiteKernel: rank(x) should be static and >= 2'),
                      tf.assert_greater_equal(tf.rank(y), 2, message='WhiteKernel: rank(y) should be static and >= 2'),
                      tf.assert_equal(tf.shape(x)[:-2], tf.shape(y)[:-2],
                                      message='WhiteKernel: shapes of x and y should match')]
        with tf.control_dependencies(assert_ops):
            shape = list(x.get_shape()[:-2]) + [self._kernel_num, x.get_shape()[-2], y.get_shape()[-2]]
            return self._variance * tf.ones(shape, dtype=self._dtype)
