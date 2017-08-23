#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np


__all__ = [
    'log_combination',
    'explicit_broadcast',
    'maybe_explicit_broadcast',
    'is_same_dynamic_shape',
]


def log_combination(n, ks):
    """
    Compute the log combination function.

    .. math::

        \\log \\binom{n}{k_1, k_2, \\dots} = \\log n! - \\sum_{i}\\log k_i!

    :param n: A N-D `float` Tensor. Can broadcast to match `ks[:-1]`.
    :param ks: A (N + 1)-D `float` Tensor. Each slice `[i, j, ..., k, :]` is
        a vector of `[k_1, k_2, ...]`.

    :return: A N-D Tensor of type same as `n`.
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


def assert_same_dtype(tensors_with_name, dtype=None):
    """
    Whether all types of tensors in `tensors` are the same as `dtype`.

    :param tensors_with_name: A list of (tensor, tensor_name).
    :param dtype: Expected type. If `None`, depend on the type of tensors.
    :return: The type of `tensors`.
    """

    expected_dtype = dtype
    for tensor, tensor_name in tensors_with_name:
        tensor_dtype = tensor.dtype
        if not expected_dtype:
            expected_dtype = tensor_dtype
        elif expected_dtype != tensor_dtype:
            if dtype is None:
                tensor0, tensor0_name = tensors_with_name[0]
                raise TypeError(
                    '%s(%s), must be the same type as %s(%s).' % (
                        tensor_name, tensor_dtype,
                        tensor0_name, tensor0.dtype))
            else:
                raise TypeError(
                    '%s(%s), must be %s.' % (
                        tensor_name, tensor_dtype, expected_dtype))

    return expected_dtype


def assert_same_specific_dtype(tensors_with_name, dtypes):
    """
    Whether all types of tensors in `tensors` are the same and in `dtypes`.

    :param tensors_with_name: A list of (tensor, tensor_name).
    :param dtypes: A list of types.
    :return: The type of `tensors`.
    """
    if tensors_with_name is None:
        return None
    tensors_dtype = assert_same_dtype(tensors_with_name)
    if tensors_dtype is not None and tensors_dtype not in dtypes:
        tensor0, tensor0_name = tensors_with_name[0]
        raise TypeError('%s(%s), must be in %s.' % (
            tensor0_name, tensor0.dtype, dtypes))
    return tensors_dtype


def assert_same_float_dtype(tensors_with_name, dtype=None):
    """
    Whether all types of tensors in `tensors` are the same and floating type.

    :param tensors_with_name: A list of (tensor, tensor_name).
    :param dtype: Expected type. If `None`, depend on the type of tensors.
    :return: The type of `tensors`.
    """

    floating_types = [tf.float16, tf.float32, tf.float64]
    if dtype is None:
        return assert_same_specific_dtype(tensors_with_name, floating_types)
    elif dtype in floating_types:
        return assert_same_dtype(tensors_with_name, dtype)
    else:
        raise TypeError("The argument 'dtype' must be in %s" % floating_types)


def assert_same_float_and_int_dtype(tensors_with_name, dtype=None):
    """
    Whether all types of tensors in `tensors` are the same and floating (or
    integer) type.

    :param tensors_with_name: A list of (tensor, tensor_name).
    :param dtype: Expected type. If `None`, depend on the type of tensors.
    :return: The type of `tensors`.
    """

    available_types = [tf.float16, tf.float32, tf.float64,
                       tf.int16, tf.int32, tf.int64]
    if dtype is None:
        return assert_same_specific_dtype(tensors_with_name, available_types)
    elif dtype in available_types:
        return assert_same_dtype(tensors_with_name, dtype)
    else:
        raise TypeError("The argument 'dtype' must be in %s" % available_types)


def assert_rank_at_least_one(tensor, name):
    """
    Whether the rank of `tensor` is at least one.

    :param tensor: A tensor to be checked.
    :param name: The name of `tensor` for error message.
    :return: (checked tensor, the last dimension of `tensor`).
    """
    static_shape = tensor.get_shape()
    shape_err_msg = name + " should have rank >= 1."
    if static_shape and (static_shape.ndims < 1):
        raise ValueError(shape_err_msg)
    elif static_shape and (static_shape[-1].value is not None):
        return tensor, static_shape[-1].value
    else:
        _assert_shape_op = tf.assert_rank_at_least(
            tensor, 1, message=shape_err_msg)
        with tf.control_dependencies([_assert_shape_op]):
            tensor = tf.identity(tensor)
        return tensor, tf.shape(tensor)[-1]


def assert_scalar(tensor, name):
    """
    Whether the `tensor` is a scalar (0-D tensor).

    :param tensor: A tensor to be checked.
    :param name: The name of `tensor` for error message.
    :return: The checked tensor.
    """
    static_shape = tensor.get_shape()
    shape_err_msg = name + " should be a scalar (0-D tensor)."
    if static_shape and (static_shape.ndims >= 1):
        raise ValueError(shape_err_msg)
    else:
        _assert_shape_op = tf.assert_rank(tensor, 0, message=shape_err_msg)
        with tf.control_dependencies([_assert_shape_op]):
            tensor = tf.identity(tensor)
        return tensor


def assert_positive_integer(value, dtype, name):
    """
    Whether `value` is a scalar (or 0-D tensor) and positive.
    If `value` is the instance of built-in type, it will be checked
    directly. Otherwise, it will be converted to a `dtype` tensor and checked.

    :param value: The value to be checked.
    :param dtype: The tensor dtype.
    :param name: The name of `value` used in error message.
    :return: The checked value.
    """
    sign_err_msg = name + " must be positive"
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(sign_err_msg)
        return value
    else:
        try:
            tensor = tf.convert_to_tensor(value, dtype)
        except ValueError:
            raise TypeError(name + ' must be ' + str(dtype))
        _assert_rank_op = tf.assert_rank(
            tensor, 0,
            message=name + " should be a scalar (0-D Tensor).")
        _assert_positive_op = tf.assert_greater(
            tensor, tf.constant(0, dtype), message=sign_err_msg)
        with tf.control_dependencies([_assert_rank_op,
                                      _assert_positive_op]):
            tensor = tf.identity(tensor)
        return tensor


def assert_positive_int32_integer(value, name):
    """
    Whether `value` is a scalar (or 0-D tensor) and positive.
    If `value` is the instance of int, it will be checked directly.
    Otherwise, it will be converted to a `int32` tensor and checked.

    :param value: The value to be checked.
    :param name: The name of `value` used in error message.
    :return: The checked value.
    """
    if isinstance(value, float):
        raise TypeError(name + ' must be an integer.')
    return assert_positive_integer(value, tf.int32, name)


def open_interval_standard_uniform(shape, dtype):
    """
    Return samples from uniform distribution in unit open interval (0, 1).

    :param shape: The shape of generated samples.
    :param dtype: The dtype of generated samples.
    :return: A Tensor of samples.
    """
    return tf.random_uniform(
        shape=shape,
        minval=np.finfo(dtype.as_numpy_dtype).tiny,
        maxval=1.,
        dtype=dtype)
