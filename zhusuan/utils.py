#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from functools import wraps

import tensorflow as tf


__all__ = [
    'TensorArithmeticMixin',
    'log_mean_exp',
    'merge_dicts',
]


class TensorArithmeticMixin(object):
    """
    Mixin class for implementing tensor arithmetic operations.

    The derived class must support `tf.convert_to_tensor`, in order to
    inherit from this mixin class.
    """

    # overloading arithmetic operations
    def __abs__(self):
        return tf.abs(self)

    def __neg__(self):
        return tf.negative(self)

    def __add__(self, other):
        return tf.add(self, other)

    def __radd__(self, other):
        return tf.add(other, self)

    def __sub__(self, other):
        return tf.subtract(self, other)

    def __rsub__(self, other):
        return tf.subtract(other, self)

    def __mul__(self, other):
        return tf.multiply(self, other)

    def __rmul__(self, other):
        return tf.multiply(other, self)

    def __div__(self, other):
        return tf.div(self, other)

    def __rdiv__(self, other):
        return tf.div(other, self)

    def __truediv__(self, other):
        return tf.truediv(self, other)

    def __rtruediv__(self, other):
        return tf.truediv(other, self)

    def __floordiv__(self, other):
        return tf.floordiv(self, other)

    def __rfloordiv__(self, other):
        return tf.floordiv(other, self)

    def __mod__(self, other):
        return tf.mod(self, other)

    def __rmod__(self, other):
        return tf.mod(other, self)

    def __pow__(self, other):
        return tf.pow(self, other)

    def __rpow__(self, other):
        return tf.pow(other, self)

    # logical operations
    def __invert__(self):
        return tf.logical_not(self)

    def __and__(self, other):
        return tf.logical_and(self, other)

    def __rand__(self, other):
        return tf.logical_and(other, self)

    def __or__(self, other):
        return tf.logical_or(self, other)

    def __ror__(self, other):
        return tf.logical_or(other, self)

    def __xor__(self, other):
        return tf.logical_xor(self, other)

    def __rxor__(self, other):
        return tf.logical_xor(other, self)

    # boolean operations
    def __lt__(self, other):
        return tf.less(self, other)

    def __le__(self, other):
        return tf.less_equal(self, other)

    def __gt__(self, other):
        return tf.greater(self, other)

    def __ge__(self, other):
        return tf.greater_equal(self, other)

    # slicing and indexing
    def __getitem__(self, item):
        return (tf.convert_to_tensor(self))[item]

    # identification
    def __hash__(self):
        # Necessary to support Python's collection membership operators
        return id(self)

    def __eq__(self, other):
        # Necessary to support Python's collection membership operators
        return id(self) == id(other)

    # disallowed operators
    def __iter__(self):
        raise TypeError(
            "{} object is not iterable.".format(self.__class__.__name__))

    def __bool__(self):
        raise TypeError(
            "Using a `{}` object as a Python `bool` is not allowed. "
            "Use `if t is not None:` instead of `if t:` to test if a "
            "tensor is defined, and use TensorFlow ops such as "
            "tf.cond to execute subgraphs conditioned on the value of "
            "a tensor.".format(self.__class__.__name__)
        )

    def __nonzero__(self):
        raise TypeError(
            "Using a `{}` object as a Python `bool` is not allowed. "
            "Use `if t is not None:` instead of `if t:` to test if a "
            "tensor is defined, and use TensorFlow ops such as "
            "tf.cond to execute subgraphs conditioned on the value of "
            "a tensor.".format(self.__class__.__name__)
        )


def log_sum_exp(x, axis=None, keepdims=False):
    """
    Deprecated: Use tf.reduce_logsumexp().

    Tensorflow numerically stable log sum of exps across the `axis`.

    :param x: A Tensor.
    :param axis: An int or list or tuple. The dimensions to reduce.
        If `None` (the default), reduces all dimensions.
    :param keepdims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.

    :return: A Tensor after the computation of log sum exp along given axes of
        x.
    """
    x = tf.convert_to_tensor(x)
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    ret = tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis,
                               keepdims=True)) + x_max
    if not keepdims:
        ret = tf.reduce_sum(ret, axis=axis)
    return ret


def log_mean_exp(x, axis=None, keepdims=False):
    """
    Tensorflow numerically stable log mean of exps across the `axis`.

    :param x: A Tensor.
    :param axis: An int or list or tuple. The dimensions to reduce.
        If `None` (the default), reduces all dimensions.
    :param keepdims: Bool. If true, retains reduced dimensions with length 1.
        Default to be False.

    :return: A Tensor after the computation of log mean exp along given axes of
        x.
    """
    x = tf.convert_to_tensor(x)
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    ret = tf.log(tf.reduce_mean(tf.exp(x - x_max), axis=axis,
                                keepdims=True)) + x_max
    if not keepdims:
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
