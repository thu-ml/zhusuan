#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


__all__ = [
    'log_factorial',
    'log_combination',
    'explicit_broadcast',
    'is_same_dynamic_shape',
]


def log_factorial(n):
    """
    Compute the log factorial function.

    .. math:: \\log n!

    :param n: A Tensor of type `int32`.

    :return: A `float32` Tensor of the same shape as `n`.
    """
    n = tf.convert_to_tensor(n, tf.int32)
    return tf.lgamma(tf.to_float(n + 1))


def log_combination(n, ks):
    """
    Compute the log combination function.

    .. math::

    \\log \binom{n}{k_1, k_2, \dots} = \\log n! - \sum_{i}\\log k_i!

    :param n: A N-D Tensor of type `int32`. Can broadcast to match `ks`[:-1].
    :param ks: A (N + 1)-D Tensor of type `int32`. Each slice
        [i, j, ..., k, :] is a vector of [k_1, k_2, ...].

    :return: A N-D Tensor of type `float32`.
    """
    n = tf.convert_to_tensor(n, tf.int32)
    ks = tf.convert_to_tensor(ks, tf.int32)
    return log_factorial(n) - tf.reduce_sum(log_factorial(ks), axis=-1)


def explicit_broadcast(x, y, x_name, y_name):
    """
    Explicit broadcast two Tensors to have the same shape.

    :return: x, y after broadcast.
    """
    try:
        x *= tf.ones_like(y)
        y *= tf.ones_like(x)
    except ValueError:
        raise ValueError(
            "{} and {} cannot broadcast to have the same shape. ("
            "{} vs. {})".format(x_name, y_name,
                                x.get_shape(), y.get_shape()))
    return x, y


def is_same_dynamic_shape(x, y):
    """
    Whether `x` and `y` has the same dynamic shape.

    :param x: A Tensor.
    :param y: A Tensor.
    :return: A scalar Tensor of `bool`.
    """
    # There is a BUG of Tensorflow for not doing static shape inference
    # right in nested tf.cond()'s, so we are not comparing x and y's
    # shape directly but working with their concatenations.
    return tf.cond(
        tf.equal(tf.rank(x), tf.rank(y)),
        lambda: tf.reduce_all(tf.equal(
            tf.concat([tf.shape(x), tf.shape(y)], 0),
            tf.concat([tf.shape(y), tf.shape(x)], 0))),
        lambda: tf.convert_to_tensor(False, tf.bool))
