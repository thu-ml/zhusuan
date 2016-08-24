#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import itertools

import tensorflow as tf
import six

from .utils import ensure_dim_match
from .layers import Discrete, get_output


def advi(model, observed_inputs, observed_layers, latent_layers,
         optimizer=tf.train.AdamOptimizer()):
    """
    Implements the automatic differentiation variational inference (ADVI)
    algorithm. For now we assume all latent variables have been transformed in
    the model definition to have their support on R^n.

    :param model: A model object that has a method logprob(latent, observed)
        to compute the log joint likelihood of the model.
    :param observed_inputs: A dictionary. Given inputs to the observed layers.
    :param observed_layers: A dictionary. The observed layers.
    :param latent_layers: A dictionary. The latent layers.
    :param optimizer: Tensorflow optimizer object. Default to be
        AdamOptimizer.

    :return: Tensorflow gradients that can be applied using
        `tf.train.Optimizer.apply_gradients`
    :return: A 0-D Tensor. The variational lower bound.
    """
    if list(six.iterkeys(observed_inputs)) != list(
            six.iterkeys(observed_layers)):
        raise ValueError("Observed layers and inputs don't match.")

    # add all observed (variable, input) pairs into inputs
    inputs = {}
    for name, layer in six.iteritems(observed_layers):
        inputs[layer] = observed_inputs[name]

    # get discrete latent layers
    latent_k, latent_v = map(list, zip(*six.iteritems(latent_layers)))
    discrete_latent_layers = dict(filter(lambda x: isinstance(x[1], Discrete),
                                         six.iteritems(latent_layers)))

    if discrete_latent_layers:
        # Discrete latent layers exists
        discrete_latent_k, discrete_latent_v = map(list, zip(
            *six.iteritems(discrete_latent_layers)))

        # get all configurations of discrete latent variables
        all_disc_latent_configs = []
        for layer in discrete_latent_v:
            tmp = []
            for i in range(layer.n_classes):
                layer_input = tf.expand_dims(tf.expand_dims(tf.one_hot(
                    i, depth=layer.n_classes, dtype=tf.float32), 0), 0)
                tmp.append(layer_input)
            all_disc_latent_configs.append(tmp)
        # cartesian products
        all_disc_latent_configs = itertools.product(*all_disc_latent_configs)

        # feed all configurations of inputs
        weighted_lbs = []
        for discrete_latent_inputs in all_disc_latent_configs:
            _inputs = inputs.copy()
            discrete_latent_inputs = dict(zip(discrete_latent_v,
                                              discrete_latent_inputs))
            _inputs.update(discrete_latent_inputs)
            # TODO: assert n_samples == 1 for discrete layers (run time)
            # ensure the batch_size dimension matches
            _inputs_k, _inputs_v = zip(*six.iteritems(_inputs))
            _inputs_v = ensure_dim_match(_inputs_v, 0)
            _inputs = dict(zip(_inputs_k, _inputs_v))
            # size: continuous layers (batch_size, n_samples, n_dim)
            #       discrete layers (batch_size, 1, n_dim)
            outputs = get_output(latent_v, _inputs)
            latent_outputs = dict(zip(latent_k, map(lambda x: x[0], outputs)))
            # size: continuous layer (batch_size, n_samples)
            #       discrete layers (batch_size, 1)
            latent_logpdfs = dict(zip(latent_k, map(lambda x: x[1], outputs)))
            # size: (batch_size, n_samples)
            lower_bound = model.log_prob(latent_outputs, observed_inputs) - \
                sum(six.itervalues(latent_logpdfs))
            discrete_latent_logpdfs = [latent_logpdfs[i]
                                       for i in discrete_latent_k]
            w = tf.exp(sum(discrete_latent_logpdfs))
            weighted_lbs.append(lower_bound * w)
        # size: (batch_size, n_samples)
        lower_bound = sum(weighted_lbs)
    else:
        # no Discrete latent layers
        outputs = get_output(latent_v, inputs)
        latent_outputs = dict(zip(latent_k, map(lambda x: x[0], outputs)))
        latent_logpdfs = map(lambda x: x[1], outputs)
        lower_bound = model.log_prob(latent_outputs, observed_inputs) - \
            sum(latent_logpdfs)

    lower_bound = tf.reduce_mean(lower_bound)
    return optimizer.compute_gradients(-lower_bound), lower_bound
