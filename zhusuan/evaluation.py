#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import six
from six.moves import zip, map
import tensorflow as tf
import numpy as np

from .utils import log_mean_exp, merge_dicts


__all__ = [
    'is_loglikelihood',
    'BDMC'
]


def is_loglikelihood(log_joint, observed, latent, axis=0):
    """
    Marginal log likelihood (:math:`\log p(x)`) estimates using self-normalized
    importance sampling.

    :param log_joint: A function that accepts a dictionary argument of
        (str, Tensor) pairs, which are mappings from all `StochasticTensor`
        names in the model to their observed values. The function should
        return a Tensor, representing the log joint likelihood of the model.
    :param observed: A dictionary of (str, Tensor) pairs. Mapping from names
        of observed `StochasticTensor` s to their values, for which to
        calculate marginal log likelihood.
    :param latent: A dictionary of (str, (Tensor, Tensor)) pairs. Mapping
        from names of latent `StochasticTensor` s to their samples and log
        probabilities.
    :param axis: The sample dimension(s) to reduce when computing the
        log likelihood.

    :return: A Tensor. The estimated log likelihood of observed data.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    log_w = log_joint(joint_obs) - sum(latent_logpdfs)
    return log_mean_exp(log_w, axis)


class BDMC:
    """
    Bidirectional Monte Carlo. Estimates a stochastic lower bound and upper
    bound of the marginal log likelihood using annealed importance sampling
    (AIS).
    """
    def __init__(self, log_prior, log_joint, prior_sampler,
                 hmc, observed, latent, chain_axis,
                 num_chains, num_temperatures):
        # Shape of latent: [chain_axis * num_data, data dims]
        # Construct the tempered objective
        self.prior_sampler = prior_sampler
        self.latent = latent
        self.chain_axis = chain_axis
        self.num_chains = num_chains
        self.num_temperatures = num_temperatures

        with tf.name_scope("BDMC"):
            self.temperature = tf.placeholder(tf.float32, shape=[], name="temperature")

            def log_fn(observed):
                return log_prior(observed) * (1 - self.temperature) + \
                       log_joint(observed) * self.temperature

            self.log_fn = log_fn
            self.log_fn_val = log_fn(merge_dicts(observed, latent))
            self.sampler = hmc.sample(log_fn, observed, latent, chain_axis=chain_axis)

            self.init_latent = [tf.assign(z, z_s)
                                for z, z_s in zip(latent.values(), self.prior_sampler.values())]

    def run(self, sess, feed_dict):
        # Draw a sample from the prior
        sess.run(self.init_latent, feed_dict=feed_dict)
        prior_density = sess.run(self.log_fn_val,
                                 feed_dict=merge_dicts(feed_dict, {self.temperature: 0}))
        log_weights = -self.sum_density(prior_density)

        # Forward AIS
        for num_t in range(self.num_temperatures):
            current_temperature = 1.0 / self.num_temperatures * (num_t + 1)
            new_feed_dict = feed_dict.copy()
            new_feed_dict[self.temperature] = current_temperature
            _, _, _, _, oldp, newp, acc, ss = sess.run(self.sampler,
                                                       feed_dict=new_feed_dict)
            oldp = self.sum_density(oldp)
            newp = self.sum_density(newp)
            if num_t + 1 < self.num_temperatures:
                log_weights += oldp - newp
            else:
                log_weights += oldp

            # print('Temperature = {}, OldP = {}, NewP = {}, Acc = {}, SS = {}'
            #       .format(current_temperature, np.mean(oldp), np.mean(newp), np.mean(acc), np.mean(ss)))

        ll_lb = np.mean(self.get_lower_bound(log_weights))

        # Backward AIS
        log_weights = -newp
        for num_t in range(self.num_temperatures):
            current_temperature = 1.0 - 1.0 / self.num_temperatures * (num_t + 1)
            _, _, _, _, oldp, newp, acc, ss = sess.run(self.sampler,
                                   feed_dict=merge_dicts(feed_dict,
                                                         {self.temperature: current_temperature}))
            oldp = self.sum_density(oldp)
            newp = self.sum_density(newp)
            if num_t + 1 < self.num_temperatures:
                log_weights += oldp - newp
            else:
                log_weights += oldp

            # print('Temperature = {}, OldP = {}, NewP = {}, Acc = {}, SS = {}'
            #       .format(current_temperature, np.mean(oldp), np.mean(newp), np.mean(acc), np.mean(ss)))

        ll_ub = -np.mean(self.get_lower_bound(log_weights))

        return ll_lb, ll_ub

    def sum_density(self, density):
        return np.reshape(density, [self.num_chains, -1])

    def get_lower_bound(self, log_weights):
        max_log_weights = np.max(log_weights, axis=0)
        offset_log_weights = np.sum(np.exp(log_weights - max_log_weights), axis=0)
        log_weights = np.log(offset_log_weights) + max_log_weights - np.log(self.num_chains)

        return log_weights