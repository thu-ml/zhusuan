#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import six
from six.moves import zip, map
import tensorflow as tf
import numpy as np

from zhusuan.utils import log_mean_exp, merge_dicts


__all__ = [
    'is_loglikelihood',
]


def is_loglikelihood(log_joint, observed, latent, axis=None):
    """
    Marginal log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param log_joint: A function that accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        `StochasticTensor` names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed `StochasticTensor` s to their values
    :param latent: A dictionary of ``(string, (Tensor, Tensor))`` pairs.
        Mapping from names of latent `StochasticTensor` s to their samples and
        log probabilities.
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in the importance sampling estimation. If None, no
        dimension is reduced.

    :return: A Tensor. The estimated log likelihood of observed data.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    log_w = log_joint(joint_obs) - sum(latent_logpdfs)
    if axis is not None:
        return log_mean_exp(log_w, axis)
    return log_w


class BDMC:
    """
    Marginal log likelihood estimates using Bidirectional Monte Carlo (Grosse,
    2015), which estimates a stochastic lower bound and a stochastic upper
    bound of the marginal log likelihood using annealed importance sampling
    (AIS).
    """
    def __init__(self, log_prior, log_joint, prior_sampler,
                 hmc, observed, latent, n_chains, n_temperatures):
        # Shape of latent: [chain_axis * num_data, data dims]
        # Construct the tempered objective
        self.prior_sampler = prior_sampler
        self.latent = latent
        self.n_chains = n_chains
        self.n_temperatures = n_temperatures

        with tf.name_scope("BDMC"):
            self.temperature = tf.placeholder(tf.float32, shape=[],
                                              name="temperature")

            def log_fn(observed):
                return log_prior(observed) * (1 - self.temperature) + \
                       log_joint(observed) * self.temperature

            self.log_fn = log_fn
            self.log_fn_val = log_fn(merge_dicts(observed, latent))
            self.sample_op, self.hmc_info = hmc.sample(
                log_fn, observed, latent)
            self.init_latent = [tf.assign(z, z_s)
                                for z, z_s in zip(latent.values(),
                                                  self.prior_sampler.values())]

    def run(self, sess, feed_dict):
        # Draw a sample from the prior
        sess.run(self.init_latent, feed_dict=feed_dict)
        prior_density = sess.run(self.log_fn_val,
                                 feed_dict=merge_dicts(
                                     feed_dict, {self.temperature: 0}))
        log_weights = -prior_density

        # Forward AIS
        for num_t in range(self.n_temperatures):
            current_temperature = 1.0 / self.n_temperatures * (num_t + 1)
            new_feed_dict = feed_dict.copy()
            new_feed_dict[self.temperature] = current_temperature
            _, old_log_p, new_log_p = sess.run(
                [self.sample_op, self.hmc_info.orig_log_prob,
                 self.hmc_info.log_prob], feed_dict=new_feed_dict)
            if num_t + 1 < self.n_temperatures:
                log_weights += old_log_p - new_log_p
            else:
                log_weights += old_log_p

        ll_lb = np.mean(self.get_lower_bound(log_weights))

        # Backward AIS
        log_weights = -new_log_p
        for num_t in range(self.n_temperatures):
            current_temperature = 1.0 - 1.0 / self.n_temperatures * (num_t + 1)
            _, old_log_p, new_log_p = sess.run(
                [self.sample_op, self.hmc_info.orig_log_prob,
                 self.hmc_info.log_prob],
                feed_dict=merge_dicts(feed_dict,
                                      {self.temperature: current_temperature}))
            if num_t + 1 < self.n_temperatures:
                log_weights += old_log_p - new_log_p
            else:
                log_weights += old_log_p

        ll_ub = -np.mean(self.get_lower_bound(log_weights))

        return ll_lb, ll_ub

    def get_lower_bound(self, log_weights):
        max_log_weights = np.max(log_weights, axis=0)
        offset_log_weights = np.sum(np.exp(log_weights - max_log_weights),
                                    axis=0)
        log_weights = np.log(offset_log_weights) + max_log_weights - \
            np.log(self.n_chains)
        return log_weights
