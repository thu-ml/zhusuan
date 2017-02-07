#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import six
from six.moves import zip, map

from .utils import log_mean_exp, merge_dicts


__all__ = [
    'is_loglikelihood',
]


def is_loglikelihood(log_joint, observed, latent, axis=0, given=None):
    """
    Data log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param log_joint: A function that accepts two arguments:
        * joint_obs: A dictionary of (str, Tensor) pairs. Mapping from
            all StochasticTensor names to their observed values.
        * given: See the `given` param.
        Represents the log joint likelihood of the model.
    :param observed: A dictionary of (str, Tensor) pairs. Given inputs to
        the observed variables.
    :param latent: A dictionary of (str, (Tensor, Tensor)) pairs. The
        value of two Tensors represents (output, logpdf) given by the
        `zhusuan.layers.get_output` function for distribution layers.
    :param axis: The sample dimension(s) to reduce when computing the
        variational lower bound.
    :param given: A dictionary of (str, Tensor) pairs. This is used when
        some deterministic transformations have been computed in the latent
        proposal and can be reused when evaluating model joint log likelihood.
        This dictionary will be directly passed to the model object.

    :return: A Tensor. The estimated log likelihood of observed data.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    given = given if given is not None else {}
    joint_obs = merge_dicts(observed, latent_outputs)
    log_w = log_joint(joint_obs, given) - sum(latent_logpdfs)
    return log_mean_exp(log_w, axis)
