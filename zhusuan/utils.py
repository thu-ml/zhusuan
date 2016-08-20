#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from functools import reduce, wraps


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


def ensure_dim_match(inputs, dim):
    """
    Ensure the specified dimension of given inputs match each other.
    Specifically, the values in this dimension can only be of the same number
    or 1, otherwise raise. The inputs with value 1 in this dimension will be
    tiled to have the same value with others.

    :param inputs: A list of Tensors.
    :param dim: Int. The dimension along which to ensure matches.
    :return: A list of Tensors that have been tiled to match.
    """
    if len(inputs) == 1:
        return inputs
    max_value = reduce(tf.maximum, [tf.shape(i)[dim] for i in inputs])
    static_max_value = None
    static_values = [i.get_shape().as_list()[dim] for i in inputs]
    if None not in static_values:
        static_max_value = max(static_values)
    ret = []
    for input_ in inputs:
        assert_op = tf.Assert(tf.logical_or(
            tf.equal(tf.shape(input_)[dim], 1),
            tf.equal(tf.shape(input_)[dim], max_value)), inputs)
        with tf.control_dependencies([assert_op]):
            input_ = tf.identity(input_)
        mask = tf.cast(tf.one_hot(dim, tf.rank(input_)), tf.bool)
        tile_shape = tf.select(
            mask, tf.ones([tf.rank(input_)], tf.int32) * max_value,
            tf.ones([tf.rank(input_)], tf.int32))
        output = tf.cond(tf.equal(tf.shape(input_)[dim], 1),
                         lambda: tf.tile(input_, tile_shape), lambda: input_)
        static_shape = input_.get_shape().as_list()
        static_shape[dim] = static_max_value
        output.set_shape(static_shape)
        ret.append(output)
    return ret


def add_name_scope(f):
    @wraps(f)
    def _func(*args, **kwargs):
        with tf.name_scope(args[0].__class__.__name__):
            with tf.name_scope(f.__name__):
                return f(*args, **kwargs)
    return _func
