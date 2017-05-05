#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import deque, OrderedDict

import tensorflow as tf


__all__ = [
    'get_backward_ops',
    'TensorArithmeticMixin',
]


class Context(object):
    """
    Context stack.
    """

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls, "_contexts"):
            cls._contexts = []
        return cls._contexts

    @classmethod
    def get_context(cls):
        try:
            return cls.get_contexts()[-1]
        except:
            raise RuntimeError("No contexts on the stack.")


def get_backward_ops(seed_tensors, treat_as_inputs=None):
    """
    Get backward ops from inputs to `seed_tensors` by topological order.

    :param seed_tensors: A Tensor or list of Tensors, for which to get all
        preceding Tensors.
    :param treat_as_inputs: None or a list of Tensors that is treated as
        inputs during the search (where to stop searching the backward graph).

    :return: A list of tensorflow `Operation` s in topological order.
    """
    if treat_as_inputs is None:
        treat_as_inputs = []
    treat_as_inputs = set(treat_as_inputs)
    if not isinstance(seed_tensors, (list, tuple)):
        seed_tensors = [seed_tensors]
    seed_tensors = [t for t in seed_tensors if t not in treat_as_inputs]
    seed_ops = list(OrderedDict.fromkeys(t.op for t in seed_tensors))
    q = deque(seed_ops)
    seen = set()
    done = set()
    ret = []
    while q:
        op = q[0]
        if op not in seen:
            seen.add(op)
            for tensor in reversed(op.inputs):
                if tensor not in treat_as_inputs:
                    q.appendleft(tensor.op)
            q.extendleft(reversed(op.control_inputs))
        else:
            # have seen this op before
            q.popleft()
            if op not in done:
                done.add(op)
                ret.append(op)
    return ret


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
