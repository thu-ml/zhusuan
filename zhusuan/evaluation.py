#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import six
from six.moves import zip, map

from .utils import log_mean_exp


def is_loglikelihood(model, observed, latent, reduction_indices=1):
    """
    Data log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param model: A model object that has a method logprob(latent, observed)
        to compute the log joint likelihood of the model.
    :param observed: A dictionary of (string, Tensor) pairs. Given inputs to
        the observed variables.
    :param latent: A dictionary of (string, (Tensor, Tensor)) pairs. The
        value of two Tensors represents (output, logpdf) given by the
        `zhusuan.layers.get_output` function for distribution layers.
    :param reduction_indices: The sample dimension(s) to reduce when
        computing the variational lower bound.

    :return: A Tensor. The estimated log likelihood of observed data.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    log_w = model.log_prob(latent_outputs, observed) - sum(latent_logpdfs)
    return log_mean_exp(log_w, reduction_indices)
