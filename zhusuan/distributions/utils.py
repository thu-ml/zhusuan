#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf


__all__ = [
    'log_combination',
    'explicit_broadcast',
    'maybe_explicit_broadcast',
    'is_same_dynamic_shape',
    'assert_same_dtype',
    'assert_same_specific_dtype',
    'assert_same_float_dtype',
    'assert_same_float_int_dtype',
]


def log_combination(n, ks):
    """
    Compute the log combination function.

    .. math::

        \\log \\binom{n}{k_1, k_2, \\dots} = \\log n! - \\sum_{i}\\log k_i!

    :param n: A N-D `float32` Tensor. Can broadcast to match `ks[:-1]`.
    :param ks: A (N + 1)-D `float32` Tensor. Each slice `[i, j, ..., k, :]` is 
        a vector of `[k_1, k_2, ...]`.

    :return: A N-D Tensor of type `float32`.
    """
    return tf.lgamma(n + 1) - tf.reduce_sum(tf.lgamma(ks + 1), axis=-1)


def explicit_broadcast(x, y, x_name, y_name):
    """
    Explicit broadcast two Tensors to have the same shape.

    :return: x, y after broadcast.
    """
    try:
        x *= tf.ones_like(y, dtype=x.dtype)
        y *= tf.ones_like(x, dtype=y.dtype)
    except ValueError:
        raise ValueError(
            "{} and {} cannot broadcast to match. ({} vs. {})".format(
                x_name, y_name, x.get_shape(), y.get_shape()))
    return x, y


def maybe_explicit_broadcast(x, y, x_name, y_name):
    """
    Explicit broadcast two Tensors to have the same shape if necessary.

    :return: x, y after broadcast.
    """
    if not (x.get_shape() and y.get_shape()):
        x, y = explicit_broadcast(x, y, x_name, y_name)
    else:
        if x.get_shape().ndims != y.get_shape().ndims:
            x, y = explicit_broadcast(x, y, x_name, y_name)
        elif x.get_shape().is_fully_defined() and \
                y.get_shape().is_fully_defined():
            if x.get_shape() != y.get_shape():
                x, y = explicit_broadcast(x, y, x_name, y_name)
        else:
            # Below code seems to induce a BUG when this function is
            # called in HMC. Probably due to tensorflow's not supporting
            # control flow edge from an op inside the body to outside.
            # We should further fix this.
            #
            # x, y = tf.cond(
            #     is_same_dynamic_shape(x, y),
            #     lambda: (x, y),
            #     lambda: explicit_broadcast(x, y, x_name, y_name))
            x, y = explicit_broadcast(x, y, x_name, y_name)
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


def assert_same_dtype(tensors, dtype=None):
    """
    Whether all types of tensors in `tensors` are the same as `dtype`.
    If the item in `tensors` doesn't have attribute `dtype` that is the
    instance of `tf.DType`, it will be ignored.

    :param tensors: A list of tensors.
    :param dtype: Expected type. If `None`, depend on the type of tensors.
    :return: The type of `tensors`.
    """
    expected_dtype = dtype
    for tensor in tensors:
        if tensor is not None and \
           hasattr(tensor, 'dtype') and \
           isinstance(tensor.dtype, tf.DType):
            tensor_dtype = tensor.dtype
            if not expected_dtype:
                expected_dtype = tensor_dtype
            elif expected_dtype != tensor_dtype:
                raise TypeError('%s, type=%s, must be %s.' % (
                    tensor.name if hasattr(tensor, 'name') else str(tensor),
                    tensor_dtype, expected_dtype))
    return expected_dtype


def assert_same_specific_dtype(tensors, dtypes):
    """
    Whether all types of tensors in `tensors` are the same and in `dtypes`.
    If the item in `tensors` doesn't have attribute `dtype` that is the
    instance of `tf.DType`, it will be ignored.

    :param tensors: A list of tensors.
    :param dtypes: A list of types.
    :return: The type of `tensors`.
    """
    if tensors is None: return None
    tensors_dtype = assert_same_dtype(tensors)
    if tensors_dtype is not None and tensors_dtype not in dtypes:
        for tensor in tensors:
            if tensor is not None and \
               hasattr(tensor, 'dtype') and \
               isinstance(tensor.dtype, tf.DType):
                raise TypeError('%s, type=%s, must be in %s.' % (
                    tensor.name if hasattr(tensor, 'name') else str(tensor),
                    tensor.dtype, dtypes))
    return tensors_dtype


def assert_same_float_dtype(tensors, dtype=None):
    """
    Whether all types of tensors in `tensors` are the same and floating type.
    If the item in `tensors` doesn't have attribute `dtype` that is the
    instance of `tf.DType`, it will be ignored.

    :param tensors: A list of tensors.
    :param dtype: Expected type. If `None`, depend on the type of tensors.
    :return: The type of `tensors`.
    """

    floating_types = [tf.float16, tf.float32, tf.float64]
    if dtype is None:
        return assert_same_specific_dtype(tensors, floating_types)
    elif dtype in floating_types:
        return assert_same_dtype(tensors, dtype)
    else:
        raise TypeError('dtype must be in %s' % floating_types)

def assert_same_float_int_dtype(tensors, dtype=None):
    """
    Whether all types of tensors in `tensors` are the same and floating (or
    integer) type.
    If the item in `tensors` doesn't have attribute `dtype` that is the
    instance of `tf.DType`, it will be ignored.

    :param tensors: A list of tensors.
    :param dtype: Expected type. If `None`, depend on the type of tensors.
    :return: The type of `tensors`.
    """

    available_types = [tf.float16, tf.float32, tf.float64,
                       tf.int16, tf.int32, tf.int64]
    if dtype is None:
        return assert_same_specific_dtype(tensors, available_types)
    elif dtype in available_types:
        return assert_same_dtype(tensors, dtype)
    else:
        raise TypeError('dtype must be in %s' % available_types)
