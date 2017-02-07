#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import six
from six.moves import zip, map
from tensorflow.python.training import moving_averages

from .utils import log_mean_exp, merge_dicts
from .evaluation import is_loglikelihood


__all__ = [
    'advi',
    'iwae',
    'rws',
    'nvil'
]


def advi(log_joint, observed, latent, axis=0, given=None):
    """
    Implements the automatic differentiation variational inference (ADVI)
    algorithm. For now we assume all latent variables have been transformed in
    the model definition to have their support on R^n.

    :param log_joint: A function that accepts two arguments:
        * joint_obs: A dictionary of (str, Tensor) pairs. Mapping from
            all StochasticTensor names to their observed values.
        * given: See the `given` param.
        Represents the log joint likelihood of the model.
    :param observed: A dictionary of (string, Tensor) pairs. Given inputs to
        the observed variables.
    :param latent: A dictionary of (string, (Tensor, Tensor)) pairs. The
        value of two Tensors represents (output, logpdf) given by the
        `zhusuan.layers.get_output` function for distribution layers.
    :param axis: The sample dimension(s) to reduce when computing the
        variational lower bound.
    :param given: A dictionary of (string, Tensor) pairs. This is used when
        some deterministic transformations have been computed in variational
        posterior and can be reused when evaluating model joint log likelihood.
        This dictionary will be directly passed to the model object.

    :return: A Tensor. The variational lower bound.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    given = given if given is not None else {}
    joint_obs = merge_dicts(observed, latent_outputs)
    lower_bound = log_joint(joint_obs, given) - sum(latent_logpdfs)
    lower_bound = tf.reduce_mean(lower_bound, axis)
    return lower_bound


def iwae(log_joint, observed, latent, axis=0, given=None):
    """
    Implements the importance weighted lower bound from (Burda, 2015).

    :param log_joint: A function that accepts two arguments:
        * joint_obs: A dictionary of (str, Tensor) pairs. Mapping from
            all StochasticTensor names to their observed values.
        * given: See the `given` param.
        Represents the log joint likelihood of the model.
    :param observed: A dictionary of (string, Tensor) pairs. Given inputs to
        the observed variables.
    :param latent: A dictionary of (string, (Tensor, Tensor)) pairs. The
        value of two Tensors represents (output, logpdf) given by the
        `zhusuan.layers.get_output` function for distribution layers.
    :param axis: The sample dimension(s) to reduce when computing the
        variational lower bound.
    :param given: A dictionary of (string, Tensor) pairs. This is used when
        some deterministic transformations have been computed in variational
        posterior and can be reused when evaluating model joint log likelihood.
        This dictionary will be directly passed to the model object.

    :return: A Tensor. The importance weighted lower bound.
    """
    return is_loglikelihood(log_joint, observed, latent, axis, given)


def rws(log_joint, observed, latent, axis=0, given=None):
    """
    Implements Reweighted Wake-sleep from (Bornschein, 2015).

    :param log_joint: A function that accepts two arguments:
        * joint_obs: A dictionary of (str, Tensor) pairs. Mapping from
            all StochasticTensor names to their observed values.
        * given: See the `given` param.
        Represents the log joint likelihood of the model.
    :param observed: A dictionary of (string, Tensor) pairs. Given inputs to
        the observed variables.
    :param latent: A dictionary of (string, (Tensor, Tensor)) pairs. The
        value of two Tensors represents (output, logpdf) given by the
        `zhusuan.layers.get_output` function for distribution layers.
    :param axis: The sample dimension(s) to reduce when computing the
        variational lower bound.
    :param given: A dictionary of (string, Tensor) pairs. This is used when
        some deterministic transformations have been computed in the proposal
        and can be reused when evaluating model joint log likelihood.
        This dictionary will be directly passed to the model object.

    :return: A Tensor. The cost to minimize given by Reweighted Wake-sleep.
    :return: A Tensor. Estimated log likelihoods.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    given = given if given is not None else {}
    joint_obs = merge_dicts(observed, latent_outputs)
    log_joint_value = log_joint(joint_obs, given)
    entropy = -sum(latent_logpdfs)
    log_w = log_joint_value + entropy
    log_w_max = tf.reduce_max(log_w, axis, keep_dims=True)
    w_u = tf.exp(log_w - log_w_max)
    w_tilde = tf.stop_gradient(w_u / tf.reduce_sum(w_u, axis, keep_dims=True))
    log_likelihood = log_mean_exp(log_w, axis)
    fake_log_joint_cost = -tf.reduce_sum(w_tilde * log_joint_value, axis)
    fake_proposal_cost = tf.reduce_sum(w_tilde * entropy, axis)
    cost = fake_log_joint_cost + fake_proposal_cost
    return cost, log_likelihood


def nvil(log_joint, observed, latent, baseline=None, axis=0, given=None,
         variance_normalization=False, alpha=0.8):
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    given = given if given is not None else {}
    joint_obs = merge_dicts(observed, latent_outputs)
    log_joint_value = log_joint(joint_obs, given)
    entropy = -sum(latent_logpdfs)
    l_signal = log_joint_value + entropy
    cost = 0.
    if baseline is not None:
        baseline = tf.expand_dims(baseline, axis)
        baseline_cost = 0.5 * tf.reduce_mean(tf.square(
            tf.stop_gradient(l_signal) - baseline), axis)
        l_signal = l_signal - baseline
        cost += baseline_cost

    if variance_normalization is True:
        bc = tf.reduce_mean(l_signal)
        bv = tf.reduce_mean(tf.square(l_signal - bc))
        moving_mean = tf.get_variable('moving_mean', shape=[],
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', shape=[],
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        update_mean = moving_averages.assign_moving_average(
            moving_mean, bc, decay=alpha)
        update_variance = moving_averages.assign_moving_average(
            moving_variance, bv, decay=alpha)
        l_signal = (l_signal - moving_mean) / tf.maximum(
            1., tf.sqrt(moving_variance))
        with tf.control_dependencies([update_mean, update_variance]):
            l_signal = tf.identity(l_signal)

    fake_log_joint_cost = -tf.reduce_mean(log_joint_value, axis)
    fake_variational_cost = tf.reduce_mean(
        tf.stop_gradient(l_signal) * entropy, axis)
    cost += fake_log_joint_cost + fake_variational_cost
    lower_bound = tf.reduce_mean(log_joint_value + entropy, axis)
    return cost, lower_bound
