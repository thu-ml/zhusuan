#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import deque, OrderedDict

import tensorflow as tf


__all__ = [
    'get_backward_ops',
    'reuse_variables',
    'reuse'
]


class Context(object):
    """
    Context stack.
    """

    def __init__(self):
        super(Context, self).__init__()

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
        except IndexError:
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


def reuse_variables(scope):
    """
    A decorator for transparent reuse of tensorflow
    `Variables <https://www.tensorflow.org/api_docs/python/tf/Variable>`_ in a
    function. The decorated function will automatically create variables the
    first time they are called and reuse them thereafter.

    .. note::

        This decorator is internally implemented by tensorflow's
        :func:`make_template` function. See `its doc
        <https://www.tensorflow.org/api_docs/python/tf/make_template>`_
        for requirements on the target function.

    :param scope: A string. The scope name passed to tensorflow
        `variable_scope()
        <https://www.tensorflow.org/api_docs/python/tf/variable_scope>`_.
    """
    return lambda f: tf.make_template(scope, f)


def reuse(scope):
    """
    (Deprecated) Alias of :func:`reuse_variables`.
    """
    # TODO: raise warning
    return reuse_variables(scope)
