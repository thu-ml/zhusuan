#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

from .utils import Context, get_unique_graph, get_backward_tensors


class StochasticTensor(object):
    def __init__(self):
        model = Model.get_context()
        model.add_stochastic_tensor(self)
        self.value = self.sample()

    def sample(self):
        raise NotImplementedError()

    def logpdf(self, given):
        raise NotImplementedError()


class Model(Context):
    """
    A context class supporting model construction in ZhuSuan.
    """
    def __init__(self):
        self.stochastic_tensors = OrderedDict()

    def add_stochastic_tensor(self, s_tensor):
        self.stochastic_tensors[s_tensor.value] = s_tensor

    def get_output(self, tensor_or_tensors, inputs=None):
        requested_tensors = tensor_or_tensors
        if not isinstance(tensor_or_tensors, (list, tuple)):
            requested_tensors = [tensor_or_tensors]
        requested_tensors = [t.value if isinstance(t, StochasticTensor) else t
                             for t in requested_tensors]

        if not inputs:
            ret = requested_tensors
        else:
            # inputs are observed
            if not isinstance(inputs, dict):
                raise TypeError("Inputs must be a dictionary or None.")

            graph = get_unique_graph(requested_tensors)
            control_dependencies = ge.ControlOutputs(graph)
            all_tensors = get_backward_tensors(requested_tensors,
                                               control_dependencies)
