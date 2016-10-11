#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import OrderedDict


class Layer(object):
    """
    The :class:`Layer` class represents a Layer used when building models (e.g.
    the model or the variational posterior).

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network. This design is adapted from
    `Lasagne <http://github.com/Lasagne/Lasagne>`_.

    :param incoming: A :class:`Layer` instance or `None`. The layer feeding
        into this layer.
    :param name: A string or None. An optional name to attach to this layer.
    """
    def __init__(self, incoming, name=None):
        self.input_layer = incoming
        self.name = name
        self.variables = OrderedDict()
        self.get_output_kwargs = []

    def get_output_for(self, input, **kwargs):
        """
        Propagates the given input through this layer (and only this layer).
        For neural networks layers, this is feedforward propagation.
        For sampling layers, this is drawing samples from the distribution.
        This is called by the base :meth:`zhusuan.layers.get_output()`
        to propagate some inputs through a network.

        This method should be overridden when implementing a new
        :class:`Layer` class. By default it raises `NotImplementedError`.

        :param input: A Tensor. The expression to propagate through this layer.

        :return: A Tensor. The output of this layer given the input.
        """
        raise NotImplementedError()


class MergeLayer(object):
    """
    The :class:`MergeLayer` class represents a layer that aggregates input from
    multiple layers.

    :param incomings: A list of :class:`Layer` objects feeding into this layer.
    :param name: A string or None. An optional name to attach to this layer.
    """
    def __init__(self, incomings, name=None):
        if not isinstance(incomings, list):
            raise TypeError("incomings of a MergeLayer (%s) must be of type "
                            "list." % self.__class__.__name__)
        self.input_layers = incomings
        self.name = name
        self.variables = OrderedDict()
        self.get_output_kwargs = []

    def get_output_for(self, inputs, **kwargs):
        """
        Propagates the given inputs through this layer (and only this layer).
        For neural networks layers, this is feedforward propagation.
        For sampling layers, this is drawing samples from the distribution.
        This is called by the base :meth:`zhusuan.layers.get_output()`
        to propagate some inputs through a network.

        This method should be overridden when implementing a new
        :class:`Layer` class with multiple inputs. By default it raises
        `NotImplementedError`.

        :param inputs: A list of Tensors. The expressions to propagate through
            this layer.

        :return: A Tensor. The output of this layer given the inputs.
        """
        raise NotImplementedError()


def get_all_layers(layer_or_layers, treat_as_inputs=None):
    """
    Get layers from the inputs to the given layers in topological order.

    :param layer_or_layers: A :class:`Layer` instance or list of :class:`Layer`
        instances, for which to get all preceding layers.
    :param treat_as_inputs: None or list of :class:`Layer` instances. The
        layers treated as input layers for which their preceding layers are
        not collected.

    :return: A list of :class:`Layer` instances. All layers in the graph in
        topological order.
    """
    try:
        q = deque(layer_or_layers)
    except TypeError:
        q = deque([layer_or_layers])
    seen = set()
    done = set()
    ret = []
    if treat_as_inputs is not None:
        seen.update(treat_as_inputs)
    while q:
        layer = q[0]
        if layer is None:
            q.popleft()
        elif layer not in seen:
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                q.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                q.appendleft(layer.input_layer)
        else:
            # have seen this layer before
            q.popleft()
            if layer not in done:
                done.add(layer)
                ret.append(layer)
    return ret


def get_output(layer_or_layers, inputs=None, **kwargs):
    """
    Compute the output of the network at one or more given layers. Optionally,
    one can define the input(s) to propagate through the network instead of
    using the input variable(s) associated with the network's input layer(s).

    :param layer_or_layers: A :class:`Layer` instance or list of :class:`Layer`
        instances, for which to get the output Tensors by building Tensorflow
        computation graphs consisting of forward propagation and sampling.
    :param inputs: None, Tensor, numpy array or dictionary.
        If None, uses the input variables associated with the
        :class:`InputLayer` instances.
        If Tensor, this defines the input for a single :class:`InputLayer`
        instance. Will throw a ValueError if there are multiple
        :class:`InputLayer` instances.
        If a numpy array, it will be wrapped as a constant Tensor.
        If a dictionary, any :class:`Layer` instance in the current graph
        can be mapped to a Tensor to use instead of its regular output.

    :return: A tuple or list of tuples. The outputs of given layers (
        `layer_or_layers`).

        For deterministic layers (d):
        If given replacements through the `inputs` argument (inputs[d]), the
        returned tuple is (inputs[d], None)
        If not, the returned tuple is (output_of_d, None)

        For distribution layers (r):
        If given replacements through the `inputs` argument (inputs[r]), the
        returned tuple is (inputs[r], logpdf_at_inputs[r])
        If not, the returned tuple is (samples_of_r, logpdf_at_samples_of_r)
    """
    requested_layers = layer_or_layers
    if not isinstance(layer_or_layers, (tuple, list)):
        requested_layers = [layer_or_layers]

    # track accepted kwargs used by get_output_for
    accepted_kwargs = {'deterministic'}

    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []

    def _whether_treat_as_input(layer):
        """
        Treat all deterministic layers in given inputs as inputs.
        Treat all distribution layers provided in given inputs but not in
        target layer_or_layers as inputs (stop searching the uprooting
        graph).
        """
        if not hasattr(layer, 'get_logpdf_for'):
            # deterministic layer
            return True
        elif layer in requested_layers:
            # distribution layer requested
            return False
        else:
            # distribution layer not requested
            return True

    treat_as_input = filter(_whether_treat_as_input, treat_as_input)
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, [layer.input, None])
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, [as_tensor(expr), None])
                           for layer, expr in inputs.items())
    elif inputs is not None:
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = [as_tensor(inputs), None]
    # Now check all InputLayers in layer-to-expression mapping (all_outputs)
    for layer, v in six.iteritems(all_outputs):
        if (isinstance(layer, InputLayer)) and (v[0] is None):
            raise ValueError("get_output() was called without giving an "
                             "input expression for the InputLayer %r (name is "
                             "%r). Please call it with a dictionary mapping "
                             "this layer to an input expression." %
                             (layer, layer.name))

    def _get_layer_inputs(layer):
        try:
            if isinstance(layer, MergeLayer):
                ret = [all_outputs[i][0] for i in layer.input_layers]
            else:
                ret = all_outputs[layer.input_layer][0]
        except KeyError:
            # one of the input_layer attributes must have been `None`
            raise ValueError("get_output() was called with free-floating "
                             "layer %r. Please call it with a "
                             "dictionary mapping this layer to an input "
                             "expression." % layer)
        return ret

    # update layer-to-expression mapping by propagating the inputs
    for layer in all_layers:
        # for layers having no given values
        if layer not in all_outputs:
            layer_inputs = _get_layer_inputs(layer)
            all_outputs[layer] = [None, None]
            all_outputs[layer][0] = layer.get_output_for(layer_inputs,
                                                         **kwargs)
            accepted_kwargs |= set(layer.get_output_kwargs)

    deterministic = kwargs.get('deterministic', False)
    if not deterministic:
        # generate logpdfs for requested distribution layers
        for layer in all_outputs:
            if hasattr(layer, 'get_logpdf_for') and (
                    layer in requested_layers):
                layer_inputs = _get_layer_inputs(layer)
                all_outputs[layer][1] = layer.get_logpdf_for(
                    all_outputs[layer][0], layer_inputs, **kwargs)
    else:
        all_outputs = dict((k, v[0]) for k, v in six.iteritems(all_outputs))

    # show argument suggestions
    non_existent_kwargs = set(kwargs.keys()) - accepted_kwargs
    if non_existent_kwargs:
        suggestions = []
        for kwarg in non_existent_kwargs:
            suggestion = get_close_matches(kwarg, accepted_kwargs)
            if suggestion:
                suggestions.append('%s (perhaps you meant %s)'
                                   % (kwarg, suggestion[0]))
            else:
                suggestions.append(kwarg)
        warn("get_output() was called with unused kwargs:\n\t%s"
             % "\n\t".join(suggestions), RuntimeWarning)

    # return the output(s) of the requested layer(s) only
    if isinstance(layer_or_layers, (tuple, list)):
        return [all_outputs[layer] for layer in layer_or_layers]
    else:
        return all_outputs[layer_or_layers]
