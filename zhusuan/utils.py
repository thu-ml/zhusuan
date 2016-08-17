#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


def log_sum_exp(x, reduction_indices=None, keep_dims=False):
    """
    Tensorflow numerically stable log sum of exps across the
    `reduction_indices`.

    :param x: A Tensor or numpy array.
    :param reduction_indices: An int or list or tuple. The dimensions to
        reduce. If `None` (the default), reduces all dimensions.
    :param keep_dims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.

    :return: A Tensor after the computation of log sum exp along given axes of
        x.
    """
    x = tf.cast(x, dtype=tf.float32)
    x_max = tf.reduce_max(x, reduction_indices, keep_dims=True)
    ret = tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices,
                               keep_dims=True)) + x_max
    if not keep_dims:
        ret = tf.reduce_sum(ret, reduction_indices)
    return ret


def log_mean_exp(x, reduction_indices=None, keep_dims=False):
    """
    Tensorflow numerically stable log mean of exps across the
    `reduction_indices`.

    :param x: A Tensor or numpy array.
    :param reduction_indices: An int or list or tuple. The dimensions to
        reduce. If `None` (the default), reduces all dimensions.
    :param keep_dims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.

    :return: A Tensor after the computation of log mean exp along given axes of
        x.
    """
    x = tf.cast(x, dtype=tf.float32)
    x_max = tf.reduce_max(x, reduction_indices, keep_dims=True)
    ret = tf.log(tf.reduce_mean(tf.exp(x - x_max), reduction_indices,
                                keep_dims=True)) + x_max
    if not keep_dims:
        ret = tf.reduce_mean(ret, reduction_indices)
    return ret


def as_tensor(input):
    """
    Wrap an input (python scalar or numpy array) into Tensors.

    :param input: A number, numpy array or Tensor.
    :return: A Tensor.
    """
    if isinstance(input, tf.Tensor) or isinstance(input, tf.Variable):
        return input
    else:
        try:
            return tf.constant(input)
        except Exception as e:
            raise TypeError("Input of type %s cannot be wrapped as Tensors. "
                            "(Error message: %s)" % (type(input), e))
