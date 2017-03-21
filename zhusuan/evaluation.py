#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import six
from six.moves import zip, map
import tensorflow as tf
import numpy as np

from .utils import log_mean_exp, merge_dicts


__all__ = [
    'is_loglikelihood',
    'BDMC',
    'ais_hmc'
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
            self.sample_op = hmc.sample(log_fn, observed, latent)
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
            _, _, _, _, oldp, newp, acc, ss = sess.run(self.sample_op,
                                                       feed_dict=new_feed_dict)
            if num_t + 1 < self.n_temperatures:
                log_weights += oldp - newp
            else:
                log_weights += oldp

        ll_lb = np.mean(self.get_lower_bound(log_weights))

        # Backward AIS
        log_weights = -newp
        for num_t in range(self.n_temperatures):
            current_temperature = 1.0 - 1.0 / self.n_temperatures * (num_t + 1)
            _, _, _, _, oldp, newp, acc, ss = sess.run(
                self.sample_op,
                feed_dict=merge_dicts(feed_dict,
                                      {self.temperature: current_temperature}))
            if num_t + 1 < self.n_temperatures:
                log_weights += oldp - newp
            else:
                log_weights += oldp

        ll_ub = -np.mean(self.get_lower_bound(log_weights))

        return ll_lb, ll_ub

    def get_lower_bound(self, log_weights):
        max_log_weights = np.max(log_weights, axis=0)
        offset_log_weights = np.sum(np.exp(log_weights - max_log_weights),
                                    axis=0)
        log_weights = np.log(offset_log_weights) + max_log_weights - \
            np.log(self.n_chains)
        return log_weights


def hmc(obj, latent, step_size, num_leapfrogs):
    old_obj = obj(latent)

    grad = tf.gradients(old_obj, latent)[0]
    momentum = tf.random_normal(shape=tf.shape(latent))

    current_momentum = momentum + step_size * grad / 2
    current_latent = latent
    for i in range(num_leapfrogs):
        current_latent = current_latent + step_size * current_momentum

        current_step_size = step_size if i + 1 < num_leapfrogs \
            else step_size / 2
        current_obj = obj(current_latent)
        current_momentum = current_momentum + current_step_size * \
            tf.gradients(current_obj, current_latent)[0]

    old_log_hamiltonian = old_obj - tf.reduce_sum(0.5 * tf.square(momentum),
                                                  -1)
    new_log_hamiltonian = current_obj - \
        tf.reduce_sum(0.5 * tf.square(current_momentum), -1)

    acceptance_rate = tf.minimum(1.0, tf.exp(new_log_hamiltonian -
                                             old_log_hamiltonian))
    return current_latent, old_obj, current_obj, old_log_hamiltonian, \
        new_log_hamiltonian, tf.stop_gradient(acceptance_rate)


def ais_hmc(log_prior, log_joint, prior_sampler,
            observed, step_size, num_temperature, num_leapfrogs):
    """
    Latent variable shape: chain data n_z
    log_prior, log_joint shape: chain data
    """
    temperature_gap = 1. / num_temperature

    def make_log_fn(temperature):
        def log_fn(latent):
            obs = merge_dicts(observed, {'z': latent})
            return log_prior(obs) * temperature + \
                log_joint(obs) * (1 - temperature)
        return log_fn

    def log_weight(observed):
        return temperature_gap * (log_joint(observed) - log_prior(observed))

    z = prior_sampler
    w = log_weight(merge_dicts(observed, {'z': z}))

    for i in range(1, num_temperature):
        current_temperature = 1.0 - temperature_gap * i
        new_z, oo, no, oh, nh, acc = \
            hmc(make_log_fn(current_temperature), z, step_size, num_leapfrogs)

        u01 = tf.random_uniform(shape=tf.shape(acc))
        if_accept = tf.to_float(u01 < acc)
        no = no * if_accept + oo * (1-if_accept)
        if_accept = tf.expand_dims(if_accept, 2)
        z = new_z * if_accept + z * (1-if_accept)
        w += log_weight(merge_dicts(observed, {'z': z}))

    w = log_mean_exp(w, axis=0)
    return tf.reduce_mean(w), \
           tf.reduce_mean(oo), tf.reduce_mean(no), \
           tf.reduce_mean(oh), \
           tf.reduce_mean(nh), tf.reduce_mean(acc)
