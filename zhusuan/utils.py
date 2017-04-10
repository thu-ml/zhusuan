#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from functools import wraps

import tensorflow as tf


__all__ = [
    'log_mean_exp',
    'merge_dicts',
]


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


def convert_to_int(x):
    """
    Try to convert input to type int in python.

    :param x: The input instance.
    :return: A int if succeed, else None.
    """
    if isinstance(x, int):
        return x
    return None


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
    ::

        class Foo(object):
            def foo(self):
                "Frobber"
                pass

        class Bar(Foo):
            @doc_inherit
            def foo(self):
                pass

    Now::

        Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
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
