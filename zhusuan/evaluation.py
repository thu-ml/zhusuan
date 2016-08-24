#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import six

from .utils import log_mean_exp
from .layers import get_output


def is_loglikelihood(model, observed_inputs, observed_layers, latent_layers):
    """
    Data log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param model: A model object that has a method logprob(latent, observed)
        to compute the log joint likelihood of the model.
    :param observed_inputs: A dictionary. Given inputs to the observed layers.
    :param observed_layers: A dictionary. The observed layers.
    :param latent_layers: A dictionary. The latent layers.

    :return: A Tensor of shape (batch_size,). The log likelihood of data (x).
    """
    if list(six.iterkeys(observed_inputs)) != list(
            six.iterkeys(observed_layers)):
        raise ValueError("Observed layers and inputs don't match.")

    # add all observed (variable, input) pairs into inputs
    inputs = {}
    for name, layer in six.iteritems(observed_layers):
        inputs[layer] = observed_inputs[name]

    latent_k, latent_v = map(list, zip(*six.iteritems(latent_layers)))
    outputs = get_output(latent_v, inputs)
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], outputs)))
    latent_logpdfs = map(lambda x: x[1], outputs)
    log_w = model.log_prob(latent_outputs, observed_inputs) - \
        sum(latent_logpdfs)
    return log_mean_exp(log_w, 1)
