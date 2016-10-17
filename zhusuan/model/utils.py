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


def get_backward_tensors(seed_tensors, treat_as_inputs=None,
                         control_dependencies=None):
    """
    Get backward tensors from inputs to `seed_tensors` by topological order.

    :param seed_tensors: A Tensor or list of Tensors, for which to get all
        preceding layers.
    :param treat_as_inputs: A Tensor or list of Tensors,
    :param control_dependencies: A dictionary or None. Used to explicitly add
        dependencies between Tensors.

    :return: A list of Tensors in topological order.
    """
    try:
        q = deque(seed_tensors)
    except TypeError:
        q = deque([seed_tensors])
    seen = set()
    done = set()
    # TODO
