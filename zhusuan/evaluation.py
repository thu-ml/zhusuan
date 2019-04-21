#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import zip, map, range
import tensorflow as tf
import numpy as np

from zhusuan.utils import merge_dicts
from zhusuan.variational import ImportanceWeightedObjective


__all__ = [
    'is_loglikelihood',
]


def is_loglikelihood(meta_bn, observed, latent=None, axis=None,
                     proposal=None):
    """
    Marginal log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param meta_bn: A :class:`~zhusuan.framework.meta_bn.MetaBayesianNet`
        instance or a log joint probability function.
        For the latter, it must accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        node names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed stochastic nodes to their values.
    :param latent: A dictionary of ``(string, (Tensor, Tensor))`` pairs.
        Mapping from names of latent stochastic nodes to their samples and
        log probabilities. `latent` and `proposal` are mutually exclusive.
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in the objective. If ``None``, no dimension is
        reduced.
    :param proposal: A :class:`~zhusuan.framework.bn.BayesianNet` instance
        that defines the proposal distributions of latent nodes.
        `proposal` and `latent` are mutually exclusive.

    :return: A Tensor. The estimated log likelihood of observed data.
    """
    return ImportanceWeightedObjective(
        meta_bn,
        observed,
        latent=latent,
        axis=axis,
        variational=proposal).tensor


class AIS:
    """
    Estimates a stochastic lower bound of the marginal log likelihood
    using annealed importance sampling (AIS) (Neal, 2001).

    :param meta_bn: A :class:`~zhusuan.framework.meta_bn.MetaBayesianNet`
        instance representing the model.
    :param proposal_meta_bn: A
        :class:`~zhusuan.framework.meta_bn.MetaBayesianNet` instance
        representing the proposal distributions.
    :param hmc: A :class:`~zhusuan.hmc.HMC` instance.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed stochastic nodes to their values.
    :param latent: A dictionary of ``(string, Variable)`` pairs.
        Mapping from names of latent `StochasticTensor` s to corresponding
        tensorflow Variables for storing their initial values and samples.
    :param n_temperatures: An int. Number of temperatures used in AIS.
    :param n_adapt: An int. Number of iterations for adapting HMC acceptance
        rate.
    :param verbose: A bool. Whether to print verbose information.
    """

    def __init__(self, meta_bn, proposal_meta_bn, hmc, observed, latent,
                 n_temperatures=1000, n_adapt=30, verbose=False):
        # Shape of latent: [chain_axis, num_data, data dims]
        # Construct the tempered objective
        self._n_temperatures = n_temperatures
        self._n_adapt = n_adapt
        self._verbose = verbose

        with tf.name_scope("AIS"):
            if callable(meta_bn):
                log_joint = meta_bn
            else:
                log_joint = lambda obs: meta_bn.observe(**obs).log_joint()

            latent_k, latent_v = zip(*six.iteritems(latent))

            prior_samples = proposal_meta_bn.observe().get(latent_k)
            log_prior = lambda obs: proposal_meta_bn.observe(**obs).log_joint()

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
                                for z, z_s in zip(latent_v, prior_samples)]

    def _map_t(self, t):
        return 1. / (1. + np.exp(-4 * (2 * t / self._n_temperatures - 1)))

    def _get_schedule_t(self, t):
        return (self._map_t(t) - self._map_t(0)) / (
            self._map_t(self._n_temperatures) - self._map_t(0))

    def run(self, sess, feed_dict):
        """
        Run the AIS loop.

        :param sess: A Tensorflow session.
        :param feed_dict: The `feed_dict` argument for ``session.run``.

        :return: The log marginal likelihood estimate.
        """
        # Help adapt the hmc size
        adp_num_t = 2 if self._n_temperatures > 1 else 1
        adp_t = self._get_schedule_t(adp_num_t)
        sess.run(self.init_latent, feed_dict=feed_dict)
        for i in range(self._n_adapt):
            _, acc = sess.run([self.sample_op, self.hmc_info.acceptance_rate],
                              feed_dict=merge_dicts(feed_dict,
                                                    {self.temperature: adp_t}))
            if self._verbose:
                print('Adapt iter {}, acc = {:.3f}'.format(i, np.mean(acc)))

        # Draw a sample from the prior
        sess.run(self.init_latent, feed_dict=feed_dict)
        prior_density = sess.run(self.log_fn_val,
                                 feed_dict=merge_dicts(
                                     feed_dict, {self.temperature: 0}))
        log_weights = -prior_density

        for num_t in range(self._n_temperatures):
            # current_temperature = 1.0 / self._n_temperatures * (num_t + 1)
            current_temperature = self._get_schedule_t(num_t + 1)

            _, old_log_p, new_log_p, acc = sess.run(
                [self.sample_op, self.hmc_info.orig_log_prob,
                 self.hmc_info.log_prob, self.hmc_info.acceptance_rate],
                feed_dict=merge_dicts(feed_dict,
                                      {self.temperature: current_temperature}))

            if num_t + 1 < self._n_temperatures:
                log_weights += old_log_p - new_log_p
            else:
                log_weights += old_log_p

            if self._verbose:
                print('Finished step {}, Temperature = {:.4f}, acc = {:.3f}'
                      .format(num_t + 1, current_temperature, np.mean(acc)))

        return np.mean(self._get_lower_bound(log_weights))

    def _get_lower_bound(self, log_weights):
        max_log_weights = np.max(log_weights, axis=0)
        offset_log_weights = np.mean(np.exp(log_weights - max_log_weights),
                                     axis=0)
        log_weights = np.log(offset_log_weights) + max_log_weights
        return log_weights
