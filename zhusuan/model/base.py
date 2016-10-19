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
            # TODO: logpdf
        else:
            # inputs are observed
            if not isinstance(inputs, dict):
                raise TypeError("Inputs must be a dictionary or None.")

            inputs = dict([(k.value, v)
                           if isinstance(k, StochasticTensor) else (k, v)
                           for k, v in inputs])

            def _whether_treat_as_inputs(tensor):
                """
                Treat all deterministic tensors in given inputs as inputs.
                Treat all stochastic tensors provided in given inputs but not
                in requested_tensors as inputs (stop searching the uprooting
                graph).
                """
                if tensor not in self.stochastic_tensors:
                    # deterministic tensor
                    return True
                elif tensor in requested_tensors:
                    # stochastic tensor requested
                    return False
                else:
                    # stochastic tensor not requested
                    return True

            treat_as_inputs = filter(_whether_treat_as_inputs, inputs.keys())
            graph = get_unique_graph(requested_tensors)
            all_tensors = get_backward_tensors(requested_tensors,
                                               treat_as_inputs)
            requested_tensor_set = set(requested_tensors)
            ordered_requested_tensors = [t for t in all_tensors
                                         if t in requested_tensor_set]

            # copy backward graphs of requested tensor by topological order
            copied_tensors = dict()
            for requested_t in ordered_requested_tensors:
                # for each requested tensor, get backward tensors
                _treat_as_inputs = [t for t in inputs.keys()
                                    if t is not requested_t]
                backward_tensors = get_backward_tensors(requested_t,
                                                        _treat_as_inputs)
                # copy the backward graph
                sgv = ge.make_view([t.op for t in backward_tensors])
                replacement_ts = copied_tensors.copy()
                replacement_ts.update(inputs)
                _ = replacement_ts.pop(requested_t, None)
                copied_sgv, info = ge.copy_with_input_replacements(
                    sgv, replacement_ts)

