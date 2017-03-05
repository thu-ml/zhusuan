#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from functools import reduce, wraps

import tensorflow as tf
import numpy as np
from six.moves import range


def log_sum_exp(x, axis=None, keep_dims=False):
    """
    Deprecated: Use tf.reduce_logsumexp().

    Tensorflow numerically stable log sum of exps across the `axis`.

    :param x: A Tensor or numpy array.
    :param axis: An int or list or tuple. The dimensions to reduce.
        If `None` (the default), reduces all dimensions.
    :param keep_dims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.

    :return: A Tensor after the computation of log sum exp along given axes of
        x.
    """
    x = tf.cast(x, dtype=tf.float32)
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    ret = tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis,
                               keep_dims=True)) + x_max
    if not keep_dims:
        ret = tf.reduce_sum(ret, axis=axis)
    return ret


def log_mean_exp(x, axis=None, keep_dims=False):
    """
    Tensorflow numerically stable log mean of exps across the `axis`.

    :param x: A Tensor or numpy array.
    :param axis: An int or list or tuple. The dimensions to reduce.
        If `None` (the default), reduces all dimensions.
    :param keep_dims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.

    :return: A Tensor after the computation of log mean exp along given axes of
        x.
    """
    x = tf.cast(x, dtype=tf.float32)
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    ret = tf.log(tf.reduce_mean(tf.exp(x - x_max), axis=axis,
                                keep_dims=True)) + x_max
    if not keep_dims:
        ret = tf.reduce_mean(ret, axis=axis)
    return ret


def as_tensor(input):
    """
    Deprecated: Use tf.convert_to_tensor.

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


def convert_to_int(x):
    """
    Try to convert input to type int in python.

    :param x: The input instance.
    :return: A int if succeed, else None.
    """
    try:
        return int(x)
    except TypeError:
        return None


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
        tile_shape = tf.where(
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
        with tf.name_scope(args[0].__class__.__name__ + '.' + f.__name__):
            return f(*args, **kwargs)

    return _func


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


class DocInherit(object):
    """
    doc_inherit decorator

    Usage:

    class Foo(object):
        def foo(self):
            "Frobber"
            pass

    class Bar(Foo):
        @doc_inherit
        def foo(self):
            pass

    Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):
        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):
        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find {} in parents".format(self.name))
        func.__doc__ = source.__doc__
        return func


doc_inherit = DocInherit


def copy(x):
    """
    Create a copy of a list of numpy arrays.

    :param x: A list of numpy arrays.
    :return: A copy of x.
    """
    return map(lambda x: np.copy(x), x)


class MeanStatistics:
    """
    The :class:`MeanStatistics` class supports estimating the mean of a
    series of numpy array in a online fashion.

    :param shape: The shape of the statistics.
    """

    def __init__(self, shape=None):
        self.shape = shape
        self.reset()

    def reset(self):
        """
        Reset the statistics.
        """
        if self.shape is None:
            self.x = 0
        else:
            self.x = np.zeros(self.shape)

        self.count = 0

    def add(self, y):
        """
        Add a new item.

        :param y: The item.
        """
        self.x += y
        self.count += 1

    def mean(self):
        """
        Give the mean of all currently added items.

        :return: The mean.
        """
        if self.count == 0:
            return self.x
        else:
            return self.x / self.count


class VarianceEstimator:
    """
    The :class:`VarianceEstimator` class implements the Welford estimator for
    online variance estimation.

    The estimator takes a series of items, where each item is a list of numpy
    arrays, and the estimator calculates the unbiased variance per dimension.

    :param shape: The shape of each item, which is a list of shapes for each
        numpy array.
    """

    def __init__(self, shape):
        self.shape = shape
        self.num_vars = len(shape)
        self.reset()

    def reset(self):
        """
        Reset the estimator.
        """
        self.count = 0
        self.mean = map(lambda shape: np.zeros(shape), self.shape)
        self.s = map(lambda shape: np.zeros(shape), self.shape)

    def add(self, x):
        """
        Add an item to the estimator.
        :param x: The item, which is a list of numpy arrays.
        """
        self.count += 1
        for i in range(self.num_vars):
            delta = x[i] - self.mean[i]
            self.mean[i] += delta / self.count
            self.s[i] += delta * (x[i] - self.mean[i])

    def variance(self):
        """
        Report the unbiased variance of all added items so far.

        :return: The unbiased variance per dimension.
        """
        if self.count <= 1:
            return map(lambda shape: np.zeros(shape), self.shape)
        else:
            return map(lambda x: x / (self.count - 1), self.s)


def if_raise(cond, exception):
    """
    If `cond` is true, raise `exception`. We can directly write if... raise
    in the code, but the problem is sometimes the raise is so difficult to
    trigger that we cannot come up with a test case. We use this statement to
    bypass coverage test in this case.

    :param cond: The condition.
    :param exception: The exception to trigger.
    """
    if cond:
        raise exception
