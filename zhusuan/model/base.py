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

from .utils import Context, get_backward_tensors, get_parent_tensors


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
        model = StochasticGraph.get_context()
        model.add_stochastic_tensor(self)
        self.incomings = incomings

    @property
    def value(self):
        return self.sample()

    def sample(self, **kwargs):
        """
        Get samples from the distribution.

        :return: A Tensor.
        """
        raise NotImplementedError()

    def log_p(self, given, inputs):
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

    def get_output(self, tensor_or_tensors, inputs=None):
        """
        Compute the outputs and log probability density/mass values
        at one or more nodes of the stochastic graph.

        By default, the stochastic graph samples Tensors at stochastic nodes
        and propagates them through deterministic nodes (nodes constructed by
        Tensorflow operations) to get outputs.
        Optionally, one can feed given values at any nodes to propagate
        through the graph instead of using original Tensors. This is intended
        for an "observe" action in stochastic graphs, which is a common
        scene when building probabilistic graphical models.

        :param tensor_or_tensors: A Tensor or a list of Tensors. for which to
            get the output Tensors by building Tensorflow computation graphs
            consisting of forward propagation and sampling.
        :param inputs: A dictionary or None. Any nodes in the current graph
            can be mapped to a Tensor to use instead of its regular output.

        :return: A tuple or list of tuples. The outputs and log probability
            density/mass values at `tensor_or_tensors`.

            For deterministic nodes (d):
            If given replacements through the `inputs` argument (inputs[d]),
            the returned tuple is (inputs[d], None).
            If not, the returned tuple is (output_of_d, None).

            For stochastic nodes (r):
            If given replacements through the `inputs` argument (inputs[r]),
            the returned tuple is (inputs[r], log_p_at_inputs[r])
            If not, the returned tuple is (samples_of_r,
            log_p_at_samples_of_r).
        """
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
                        output[1] = s_tensor.log_p(inputs[tensor])
                    else:
                        output[1] = s_tensor.log_p(output[0])
                ret.append(output)
        else:
            # inputs are observed
            if not isinstance(inputs, dict):
                raise TypeError("Inputs must be a dictionary or None.")

            inputs = dict([(k.value, v)
                           if isinstance(k, StochasticTensor) else (k, v)
                           for k, v in six.iteritems(inputs)])

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
            all_tensors = get_backward_tensors(requested_tensors,
                                               treat_as_inputs)
            # print(all_tensors)

            # copy each op by topological order.
            copied_tensors = {}
            copied_ops = set()

            def _replace_t_with_replacement_handler(info, t):
                if t in inputs:
                    return inputs[t]
                elif t in copied_tensors:
                    return copied_tensors[t]
                else:
                    return ge.keep_t_if_possible_handler(info, t)

            def _whether_to_copy(t):
                if t in treat_as_inputs:
                    return False
                for parent in get_parent_tensors(t, control_deps=True):
                    if parent in inputs:
                        return True
                    elif parent in copied_tensors:
                        return True
                return False

            for tensor in all_tensors:
                # print(tensor.name, _whether_to_copy(tensor))
                if (_whether_to_copy(tensor)) and (
                        tensor.op not in copied_ops):
                    sgv = ge.make_view([tensor.op])
                    copier = ge.Transformer()
                    copier.transform_external_input_handler = \
                        _replace_t_with_replacement_handler
                    copier.transform_control_input_handler = \
                        _replace_t_with_replacement_handler
                    # not changing scope for now
                    copied_sgv, info = copier(sgv, sgv.graph, "", "",
                                              reuse_dst_scope=True)
                    copied_ops.add(tensor.op)
                    op_outputs = tensor.op.outputs[:]
                    for output in op_outputs:
                        copied_tensors[output] = info.transformed(output)

            def _get_output_tensor(t):
                """
                Get corresponding output of a tensor.
                """
                if t in inputs:
                    return inputs[t]
                elif tensor in copied_tensors:
                    return copied_tensors[t]
                else:
                    return t

            # compute log probability density for each requested tensor
            ret = []
            for tensor in requested_tensors:
                output = [_get_output_tensor(tensor), None]
                if tensor in self.stochastic_tensors:
                    s_tensor = self.stochastic_tensors[tensor]
                    dist_inputs = list(map(_get_output_tensor,
                                           s_tensor.inputs))
                    output[1] = s_tensor.log_p(output[0], dist_inputs)
                ret.append(output)

        if isinstance(tensor_or_tensors, (tuple, list)):
            return ret
        else:
            return ret[0]
