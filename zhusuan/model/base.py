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
    def __init__(self, inputs):
        model = Model.get_context()
        model.add_stochastic_tensor(self)
        self.value = self.sample()
        self.inputs = inputs

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
            ret = []
            for tensor in requested_tensors:
                output = [tensor, None]
                if tensor in self.stochastic_tensors:
                    s_tensor = self.stochastic_tensors[tensor]
                    if tensor in inputs:
                        output[1] = s_tensor.logpdf(inputs[tensor],
                                                    s_tensor.inputs)
                    else:
                        output[1] = s_tensor.logpdf(output[0], s_tensor.inputs)
                ret.append(output)
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

            # copy each op's surrounding subgraph by topological order.
            copied_tensors = {}
            copied_ops = set()
            copy_info = {}
            for tensor in all_tensors:
                # TODO: control dependencies, variable reuse
                if tensor.op not in copied_ops:
                    op_inputs = tensor.op.inputs[:]
                    op_outputs = tensor.op.outputs[:]
                    replacement_ts = {}
                    for op_input in op_inputs:
                        if op_input in inputs:
                            replacement_ts[op_input] = inputs[op_input]
                        elif op_input in copied_tensors:
                            replacement_ts[op_input] = copied_tensors[op_input]
                    sgv = ge.make_view([tensor.op])
                    copied_sgv, info = ge.copy_with_input_replacements(
                        sgv, replacement_ts)
                    copied_ops.add(tensor.op)
                    for output in op_outputs:
                        copied_tensors[output] = info.transformed(output)
                        copy_info[output] = info

            # compute log probability density for each requested tensor
            ret = []
            for tensor in requested_tensors:
                output = [copied_tensors[tensor], None]
                if tensor in self.stochastic_tensors:
                    s_tensor = self.stochastic_tensors[tensor]
                    copied_inputs = [copied_tensors.get(t, t)
                                     for t in s_tensor.inputs]
                    if tensor in inputs:
                        output[1] = s_tensor.logpdf(inputs[tensor],
                                                    copied_inputs)
                    else:
                        output[1] = s_tensor.logpdf(output[0], copied_inputs)
                ret.append(output)

        return ret
