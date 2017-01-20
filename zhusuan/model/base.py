#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import OrderedDict

import six
from six.moves import map
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.python.ops import control_flow_ops

from .utils import Context, get_backward_ops


class StochasticTensor(object):
    """
    The :class:`StochasticTensor` class is the base class for various
    distributions used when building stochastic graphs. It is a wrapper
    on Tensor instances which enables transparent building of stochastic graphs
    using Tensorflow primitives.

    :param incomings: A list of Tensors. Parameters needed to specify the
        distribution.
    """
    def __init__(self, incomings):
        self.incomings = incomings
        model = StochasticGraph.get_context()
        model.add_stochastic_tensor(self)

    @property
    def value(self):
        if not hasattr(self, '_value'):
            self._value = self.sample()
        return self._value

    def sample(self, **kwargs):
        """
        Get samples from the distribution.

        :return: A Tensor.
        """
        raise NotImplementedError()

    def log_prob(self, given, inputs):
        """
        Compute log probability density (mass) function at `given` values,
        provided with parameters `inputs`.

        :param given: A Tensor. The value at which to evaluate log probability
            density (mass) function.
        :param inputs: A list of Tensors. Parameters needed to specify the
            distribution.

        :return: A Tensor. The log probability density (mass) evaluated.
        """
        raise NotImplementedError()


class StochasticGraph(Context):
    """
    A context class supporting model construction in ZhuSuan as stochastic
    graphs.
    """
    def __init__(self):
        self.stochastic_tensors = OrderedDict()

    def add_stochastic_tensor(self, s_tensor):
        """
        Add a stochastic tensor to the graph. This is the function called when
        a stochastic tensor is created in the context.

        :param s_tensor: A :class:`StochasticTensor` instance.
        """
        self.stochastic_tensors[s_tensor.value] = s_tensor
