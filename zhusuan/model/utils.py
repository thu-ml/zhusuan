#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import deque

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


def get_backward_tensors(seed_tensors, treat_as_inputs=None):
    """
    Get backward tensors from inputs to `seed_tensors` by topological order.

    :param seed_tensors: A Tensor or list of Tensors, for which to get all
        preceding Tensors.
    :param treat_as_inputs: None or a list of Tensors that is treated as
        inputs during the search (where to stop searching the backward graph).

    :return: A list of Tensors in topological order.
    """
    try:
        q = deque(seed_tensors)
    except TypeError:
        q = deque([seed_tensors])
    seen = set()
    done = set()
    ret = []
    if treat_as_inputs is not None:
        seen.update(treat_as_inputs)
    while q:
        tensor = q[0]
        if tensor not in seen:
            seen.add(tensor)
            for parent in reversed(tensor.op.inputs):
                q.appendleft(parent)
            for dep in reversed(tensor.op.control_inputs):
                q.extendleft(reversed(dep.outputs))
        else:
            # have seen this tensor before
            q.popleft()
            if tensor not in done:
                done.add(tensor)
                ret.append(tensor)
    return ret
