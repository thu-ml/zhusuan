#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = [
    'MeanFunction',
    'ConstantMeanFunction',
    'LinearMeanFunction',
]


class MeanFunction(object):
    def __init__(self, output_num):
        self._output_num = output_num

    def __call__(self, x):
        raise NotImplementedError()

    @property
    def output_num(self):
        return self._output_num


class ConstantMeanFunction(MeanFunction):
    def __init__(self, output_num, constant=0., trainable=False):
        super(ConstantMeanFunction, self).__init__(output_num)
        self._constant = constant * tf.get_variable('ConstantMeanFunction', [1], initializer=tf.zeros_initializer(),
                                                    trainable=trainable)

    def __call__(self, x):
        rank = x.shape.ndims
        return self._constant * tf.ones_like(
            tf.tile(tf.expand_dims(tf.reduce_sum(x, -1), -2), [1] * (rank - 2) + [self._output_num, 1]))

    @property
    def constant(self):
        return self._constant


class LinearMeanFunction(MeanFunction):
    def __init__(self, output_dim, weight, trainable=False):
        super(LinearMeanFunction, self).__init__(output_dim)
        self._weight = tf.Variable(weight, trainable=trainable, dtype=tf.float32)

    def __call__(self, x):
        rank = x.shape.ndims
        return tf.matrix_transpose(tf.matmul(x, tf.tile(tf.reshape(self._weight, [1] * (rank - 2) + [
            self._weight.shape[0], self._weight.shape[1]]), [tf.shape(x)[0], 1, 1])))

    @property
    def weight(self):
        return self._weight
