#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import deque, OrderedDict

import tensorflow as tf


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


def get_unique_graph(tensor_or_tensors):
    """
    Check input tensors share a unique graph and return it.

    :param tensor_or_tensors: A Tensor or a list of Tensors.
    :return: A tf.Graph instance. The unique graph shared by inputs.
    """
    if not isinstance(tensor_or_tensors, (list, tuple)):
        tensor_or_tensors = [tensor_or_tensors]
    graph = None
    for tensor in tensor_or_tensors:
        if not hasattr(tensor, 'graph'):
            raise TypeError("Inputs to get_unique_graph() are not Tensors.")
        if graph is None:
            graph = tensor.graph
        elif graph is not tensor.graph:
            raise ValueError("Tensors do not come from the same graph.")
    return graph


def get_backward_ops(seed_tensors, treat_as_inputs=None):
    """
    Get backward ops from inputs to `seed_tensors` by topological order.

    :param seed_tensors: A Tensor or list of Tensors, for which to get all
        preceding Tensors.
    :param treat_as_inputs: None or a list of Tensors that is treated as
        inputs during the search (where to stop searching the backward graph).

    :return: A list of tf.Operation's in topological order.
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
