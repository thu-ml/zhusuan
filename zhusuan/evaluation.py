#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import six
from six.moves import zip, map, range
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


class AIS:
    """
    Estimates a stochastic lower bound of the marginal log likelihood
    using annealed importance sampling (AIS).
    """
    def __init__(self, log_prior, log_joint, prior_sampler,
                 hmc, observed, latent, n_chains=25, n_temperatures=1000,
                 verbose=False):
        # Shape of latent: [chain_axis, num_data, data dims]
        # Construct the tempered objective
        self.n_chains = n_chains
        self.n_temperatures = n_temperatures
        self.verbose = verbose

        with tf.name_scope("AIS"):
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
                                                  prior_sampler.values())]

    def map_t(self, t):
        return 1. / (1. + np.exp(-4*(2*t / self.n_temperatures - 1)))

    def get_schedule_t(self, t):
        return (self.map_t(t) - self.map_t(0)) \
            / (self.map_t(self.n_temperatures) - self.map_t(0))

    def run(self, sess, feed_dict):
        # Help adapt the hmc size
        n_adp = 30
        adp_num_t = 2 if self.n_temperatures > 1 else 1
        adp_t = self.get_schedule_t(adp_num_t)
        sess.run(self.init_latent, feed_dict=feed_dict)
        for i in range(n_adp):
            _, acc = sess.run([self.sample_op, self.hmc_info.acceptance_rate],
                              feed_dict=merge_dicts(feed_dict,
                                                    {self.temperature: adp_t}))
            if self.verbose:
                print('Adapt iter {}, acc = {:.3f}'.format(i, np.mean(acc)))

        # Draw a sample from the prior
        sess.run(self.init_latent, feed_dict=feed_dict)
        prior_density = sess.run(self.log_fn_val,
                                 feed_dict=merge_dicts(
                                     feed_dict, {self.temperature: 0}))
        log_weights = -prior_density

        for num_t in range(self.n_temperatures):
            # current_temperature = 1.0 / self.n_temperatures * (num_t + 1)
            current_temperature = self.get_schedule_t(num_t + 1)

            _, old_log_p, new_log_p, acc = sess.run(
                [self.sample_op, self.hmc_info.orig_log_prob,
                 self.hmc_info.log_prob, self.hmc_info.acceptance_rate],
                feed_dict=merge_dicts(feed_dict,
                                      {self.temperature: current_temperature}))

            if num_t + 1 < self.n_temperatures:
                log_weights += old_log_p - new_log_p
            else:
                log_weights += old_log_p

            if self.verbose:
                print('Finished step {}, Temperature = {:.4f}, acc = {:.3f}'
                      .format(num_t+1, current_temperature, np.mean(acc)))

        return np.mean(self.get_lower_bound(log_weights))

    def get_lower_bound(self, log_weights):
        max_log_weights = np.max(log_weights, axis=0)
        offset_log_weights = np.sum(np.exp(log_weights - max_log_weights),
                                    axis=0)
        log_weights = np.log(offset_log_weights) + max_log_weights - \
            np.log(self.n_chains)
        return log_weights
