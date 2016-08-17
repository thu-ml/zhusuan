#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import OrderedDict, deque
from inspect import getargspec
from difflib import get_close_matches
from warnings import warn

import tensorflow as tf
import prettytensor as pt
import six

from .distributions import norm, discrete
from .utils import as_tensor


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


class InputLayer(Layer):
    """
    The :class`InputLayer` class represents a network input.

    :param shape: A tuple. The input shape.
    :param input: A Tensor or `None`, which represents a network input.
    :param name: A string or None. An optional name to attach to this layer.
    """
    def __init__(self, shape, input=None, name=None):
        super(InputLayer, self).__init__(None, name)
        if any(isinstance(s, tf.Tensor) or isinstance(s, tf.Variable)
               for s in shape):
            raise ValueError("The shape of %s is a tuple with symbolic "
                             "Tensors. Only tuples of integers and Nones "
                             "are allowed." % self.__class__.__name__)
        elif any(d is not None and d <= 0 for d in shape):
            raise ValueError((
                 "Cannot create InputLayer with a non-positive shape "
                 "dimension. shape=%r, self.name=%r") % (shape, self.name))
        if input is not None:
            input = as_tensor(input)
            if len(input.get_shape()) != len(shape):
                raise ValueError(
                    "InputLayer shape dimension mismatch: Expected shape %s, "
                    "given input shape %s" % (
                        shape, tuple(input.get_shape().as_list())))
        self.shape = shape
        self.input = input


class ReparameterizedNormal(MergeLayer):
    """
    The :class:`ReparameterizedNormal` class represents a Normal distribution
    layer that accepts the mean and the log standard deviation as inputs, which
    is used in Automatic Differentiation Variational Inference (ADVI).

    Note that gradients on samples from this Normal distribution are allowed
    to propagate into inputs in this function, using the reparametrization
    trick from (Kingma, 2013), which is contrary to the behavior in
    `zhusuan.distributions`.

    :param incomings: A list of 2 :class:`Layer` instances. The first
        representing the mean, and the second representing the log standard
        deviation. Must be of shape 3-D like (batch_size, n_samples, n_dims).
    :param n_samples: Int. Number of samples drawn for distribution layers.
        Default to be 1.
    :param name: a string or None. An optional name to attach to this layer.
    """
    def __init__(self, incomings, n_samples=1, name=None):
        super(ReparameterizedNormal, self).__init__(incomings, name)
        if len(incomings) != 2:
            raise ValueError("ReparameterizedNormal layer only accepts input "
                             "layers of length 2 (the mean and the log "
                             "standard deviation).")
        self.l_mean = incomings[0]
        self.l_logstd = incomings[1]
        self.n_samples = n_samples

    def get_output_for(self, inputs, **kwargs):
        mean_, logstd = inputs
        if self.n_samples == 1:
            samples = norm.rvs(size=tf.shape(mean_)) * tf.exp(logstd) + mean_
        else:
            samples = norm.rvs(
                size=(tf.shape(mean_)[0], self.n_samples, tf.shape(mean_)[2])
            ) * logstd + mean_
            samples.set_shape((None, self.n_samples, None))
        return samples

    def get_logpdf_for(self, output, inputs, **kwargs):
        mean_, logstd = inputs
        return tf.reduce_sum(norm.logpdf(output, mean_, logstd), 2)


class Discrete(Layer):
    """
    The :class:`Discrete` class represents a discrete distribution layer that
    accepts the class probabilities as inputs.

    :param incoming: A :class:`Layer` instance. The layer feeding into this
        layer which gives output as class probabilities. Must be of shape 3-D
        like (batch_size, n_samples, n_dims).
    :param n_samples: Int. Number of samples drawn for distribution layers.
        Default to be 1.
    :param name: a string or None. An optional name to attach to this layer.
    """
    def __init__(self, incoming, n_samples=1, name=None):
        super(Discrete, self).__init__(incoming, name)
        self.n_samples = n_samples

    def get_output_for(self, input, **kwargs):
        n_dim = tf.shape(input)[2]
        if self.n_samples == 1:
            samples_2d = discrete.rvs(
                tf.reshape(input, (-1, n_dim)))
            samples = tf.reshape(samples_2d, (-1, tf.shape(input)[1], n_dim))
            samples.set_shape(input.get_shape())
        else:
            samples_2d = discrete.rvs(
                tf.reshape(tf.tile(input, (1, self.n_samples, 1)),
                           (-1, n_dim)))
            samples = tf.reshape(samples_2d, (-1, self.n_samples, n_dim))
            samples.set_shape((input.get_shape()[0], self.n_samples,
                               input.get_shape()[2]))
        return samples

    def get_logpdf_for(self, output, input, **kwargs):
        output_2d = tf.reshape(output, (-1, tf.shape(output)[2]))
        input_2d = tf.reshape(tf.tile(input, (1, self.n_samples, 1)),
                              (-1, tf.shape(input)[2]))
        ret = tf.reshape(discrete.logpdf(output_2d, input_2d),
                         (-1, output.get_shape().as_list()[1]))
        ret.set_shape(output.get_shape()[:2])
        return ret


class PrettyTensor(MergeLayer):
    """
    The :class:`PrettyTensor` class represents a deterministic layer that
    applies a prettytensor-based deterministic transformation on the inputs.

    :param incomings: A dictionary of :class:`Layer` instances with key-value
        pairs of (name, Layer).
    :param pt_expr: A prettytensor expression that takes given names
        of the `incomings` as templates. It will be fed by the output of the
        corresponding incoming layer when calling get_output_for.
    :param name: a string or None. An optional name to attach to this layer.
    """
    def __init__(self, incomings, pt_expr, name=None):
        ks = incomings.keys()
        vs = [incomings[k] for k in incomings]
        super(PrettyTensor, self).__init__(vs, name)
        self.template_names = ks
        if not isinstance(pt_expr, pt.pretty_tensor_class._DeferredLayer):
            raise TypeError("PrettyTensor Layer only accepts pt_expr of "
                            "prettytensor template type.")
        self.pt_expr = pt_expr

    def get_output_for(self, inputs, **kwargs):
        template_mapping = dict(zip(self.template_names, inputs))
        try:
            return self.pt_expr.construct(**template_mapping).tensor
        except ValueError as e:
            raise ValueError("PrettyTensor Layer only accepts prettytensor "
                             "expression that takes names of incomings as "
                             "templates. Check the pt_expr passed on "
                             "construction. Error message: %s" % e)


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
    if not isinstance(layer_or_layers, list):
        requested_layers = [layer_or_layers]

    # # track accepted kwargs used by get_output_for
    # accepted_kwargs = set()
    #
    # def _collect_arg_suggestions(layer):
    #     ret = set()
    #     try:
    #         names, _, _, defaults = getargspec(layer.get_output_for)
    #     except TypeError:
    #         # if introspection is not possible, skip it
    #         pass
    #     else:
    #         if defaults is not None:
    #             ret |= set(names[-len(defaults):])
    #     ret |= set(layer.get_output_kwargs)
    #     return ret

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
                             "input expression for the InputLayer %r. Please "
                             "call it with a dictionary mapping this layer to "
                             "an input expression." % layer)

    def _get_layer_inputs(layer):
        print(layer.name)
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
            # accepted_kwargs |= _collect_arg_suggestions(layer)

    # generate logpdfs for requested distribution layers
    for layer in all_outputs:
        if hasattr(layer, 'get_logpdf_for') and (layer in requested_layers):
            layer_inputs = _get_layer_inputs(layer)
            all_outputs[layer][1] = layer.get_logpdf_for(
                all_outputs[layer][0], layer_inputs, **kwargs)

    # # show argument suggestions
    # non_existent_kwargs = set(kwargs.keys()) - accepted_kwargs
    # if non_existent_kwargs:
    #     suggestions = []
    #     for kwarg in non_existent_kwargs:
    #         suggestion = get_close_matches(kwarg, accepted_kwargs)
    #         if suggestion:
    #             suggestions.append('%s (perhaps you meant %s)'
    #                                % (kwarg, suggestion[0]))
    #         else:
    #             suggestions.append(kwarg)
    #     warn("get_output() was called with unused kwargs:\n\t%s"
    #          % "\n\t".join(suggestions))

    # return the output(s) of the requested layer(s) only
    if isinstance(layer_or_layers, list):
        return [all_outputs[layer] for layer in layer_or_layers]
    else:
        return all_outputs[layer_or_layers]
