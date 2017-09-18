#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import warnings

import tensorflow as tf
import six
from six.moves import zip, map
from tensorflow.python.training import moving_averages

from zhusuan.utils import log_mean_exp, merge_dicts
from zhusuan.evaluation import is_loglikelihood


__all__ = [
    'sgvb',
    'iwae',
    'rws',
    'nvil',
    'vimco'
]


def sgvb(log_joint, observed, latent, axis=None):
    """
    Implements the stochastic gradient variational bayes (SGVB) algorithm
    from (Kingma, 2013). This only works for continuous latent
    `StochasticTensor` s that can be reparameterized (Kingma, 2013).

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
        outer expectation in variational lower bound. If `None`, no dimension
        is reduced.

    :return: A Tensor. The variational lower bound.
    """
    warnings.warn("sgvb(): This function will be deprecated in the coming "
                  "version (0.3.1). Variational utilities are moving to "
                  "`zs.variational`. The new sgvb gradient estimator can be "
                  "accessed by first constructing the elbo objective (using "
                  "`zs.variational.elbo` and then calling its sgvb() method.",
                  category=FutureWarning)
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    lower_bound = log_joint(joint_obs) - sum(latent_logpdfs)
    if axis is not None:
        lower_bound = tf.reduce_mean(lower_bound, axis)
    return lower_bound


def iwae(log_joint, observed, latent, axis=None):
    """
    Implements the importance weighted lower bound from (Burda, 2015).
    This only works for continuous latent `StochasticTensor` s that
    can be reparameterized (Kingma, 2013).

    :param log_joint: A function that accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        `StochasticTensor` names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of ``(string, (Tensor, Tensor))``) pairs.
        Mapping from names of latent `StochasticTensor` s to their samples and
        log probabilities.
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in variational lower bound. If `None`, no dimension
        is reduced.

    :return: A Tensor. The importance weighted lower bound.
    """
    warnings.warn("iwae(): This function will be deprecated in the coming "
                  "version (0.3.1). Variational utilities are moving to "
                  "`zs.variational`. The new iwae gradient estimator can be "
                  "accessed by first constructing the importance weighted "
                  "objective (using `zs.variational.iw_objective` and then "
                  "calling its sgvb() method.", category=FutureWarning)
    return is_loglikelihood(log_joint, observed, latent, axis)


def rws(log_joint, observed, latent, axis=None):
    """
    Implements Reweighted Wake-sleep from (Bornschein, 2015). This works for
    both continuous and discrete latent `StochasticTensor` s.

    :param log_joint: A function that accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        `StochasticTensor` names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of ``(string, (Tensor, Tensor))``) pairs.
        Mapping from names of latent `StochasticTensor` s to their samples and
        log probabilities.
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in log likelihood and in the cost for adapting
        proposals. If `None`, no dimension is reduced.

    :return: A Tensor. The surrogate cost to minimize.
    :return: A Tensor. Estimated log likelihoods.
    """
    warnings.warn("rws(): This function will be deprecated in the coming "
                  "version (0.3.1). Variational utilities are moving to "
                  "`zs.variational`. Features of the original rws() can be "
                  "achieved by two new variational objectives. For learning "
                  "model parameters, please use the importance weighted "
                  "objective: `zs.variational.iw_objective()`. For adapting "
                  "the proposal, the new rws gradient estimator can be "
                  "accessed by first constructing the inclusive KL divergence "
                  "objective using `zs.variational.klpq` and then calling "
                  "its rws() method.", category=FutureWarning)
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    log_joint_value = log_joint(joint_obs)
    entropy = -sum(latent_logpdfs)
    log_w = log_joint_value + entropy
    if axis is not None:
        log_w_max = tf.reduce_max(log_w, axis, keep_dims=True)
        w_u = tf.exp(log_w - log_w_max)
        w_tilde = tf.stop_gradient(
            w_u / tf.reduce_sum(w_u, axis, keep_dims=True))
        log_likelihood = log_mean_exp(log_w, axis)
        fake_log_joint_cost = -tf.reduce_sum(w_tilde * log_joint_value, axis)
        fake_proposal_cost = tf.reduce_sum(w_tilde * entropy, axis)
        cost = fake_log_joint_cost + fake_proposal_cost
    else:
        cost = log_w
        log_likelihood = log_w
    return cost, log_likelihood


def nvil(log_joint,
         observed,
         latent,
         baseline=None,
         decay=0.8,
         variance_normalization=False,
         axis=None):
    """
    Implements the variance reduced score function estimator for gradients
    of the variational lower bound from (Mnih, 2014). This algorithm is also
    called "REINFORCE" or "baseline". This works for both continuous and
    discrete latent `StochasticTensor` s.

    :param log_joint: A function that accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        `StochasticTensor` names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of ``(string, (Tensor, Tensor))``) pairs.
        Mapping from names of latent `StochasticTensor` s to their samples and
        log probabilities.
    :param baseline: A Tensor that can broadcast to match the shape returned
        by `log_joint`. A trainable estimation for the scale of the
        variational lower bound, which is typically dependent on observed
        values, e.g., a neural network with observed values as inputs.
    :param variance_normalization: Whether to use variance normalization.
    :param decay: Float. The moving average decay for variance normalization.
    :param axis: The sample dimension(s) to reduce when computing the
        outer expectation in variational lower bound. If `None`, no dimension
        is reduced.

    :return: A Tensor. The surrogate cost to minimize.
    :return: A Tensor. The variational lower bound.
    """
    warnings.warn("nvil(): This function will be deprecated in the coming "
                  "version (0.3.1). Variational utilities are moving to "
                  "`zs.variational`. The new nvil gradient estimator can be "
                  "accessed by first constructing the elbo objective (using "
                  "`zs.variational.elbo` and then calling its reinforce() "
                  "method.", category=FutureWarning)
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    log_joint_value = log_joint(joint_obs)
    entropy = -sum(latent_logpdfs)
    l_signal = log_joint_value + entropy
    cost = 0.

    if baseline is not None:
        baseline_cost = 0.5 * tf.square(tf.stop_gradient(l_signal) - baseline)
        l_signal = l_signal - baseline
        cost += baseline_cost

    if variance_normalization is True:
        # TODO: extend to non-scalar
        bc = tf.reduce_mean(l_signal)
        bv = tf.reduce_mean(tf.square(l_signal - bc))
        moving_mean = tf.get_variable(
            'moving_mean', shape=[], initializer=tf.constant_initializer(0.),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', shape=[],
            initializer=tf.constant_initializer(1.), trainable=False)

        update_mean = moving_averages.assign_moving_average(
            moving_mean, bc, decay=decay)
        update_variance = moving_averages.assign_moving_average(
            moving_variance, bv, decay=decay)
        l_signal = (l_signal - moving_mean) / tf.maximum(
            1., tf.sqrt(moving_variance))
        with tf.control_dependencies([update_mean, update_variance]):
            l_signal = tf.identity(l_signal)

    fake_log_joint_cost = -log_joint_value
    fake_variational_cost = tf.stop_gradient(l_signal) * entropy
    cost += fake_log_joint_cost + fake_variational_cost
    lower_bound = log_joint_value + entropy
    if axis is not None:
        cost = tf.reduce_mean(cost, axis)
        lower_bound = tf.reduce_mean(lower_bound, axis)
    return cost, lower_bound


def vimco(log_joint, observed, latent, axis=None):
    """
    Implements the multi-sample variance reduced score function estimator for
    gradients of the variational lower bound from (Minh, 2016). This works for
    both continuous and discrete latent `StochasticTensor` s.

    .. note::

        :func:`vimco` is a multi-sample objective, size along `axis` in the
        objective should be larger than 1, else an error is raised.

    :param log_joint: A function that accepts a dictionary argument of
        ``(string, Tensor)`` pairs, which are mappings from all
        `StochasticTensor` names in the model to their observed values. The
        function should return a Tensor, representing the log joint likelihood
        of the model.
    :param observed: A dictionary of ``(string, Tensor)`` pairs. Mapping from
        names of observed `StochasticTensor` s to their values.
    :param latent: A dictionary of ``(string, (Tensor, Tensor))``) pairs.
        Mapping from names of latent `StochasticTensor` s to their samples and
        log probabilities.
    :param axis: The sample dimension to reduce when computing the
        outer expectation in variational lower bound. Must be specified. If
        `None`, an error is raised.

    :return: A Tensor. The surrogate cost to minimize.
    :return: A Tensor. The variational lower bound.
    """
    warnings.warn("vimco(): This function will be deprecated in the coming "
                  "version (0.3.1). Variational utilities are moving to "
                  "`zs.variational`. The new vimco gradient estimator can be "
                  "accessed by first constructing the importance weighted "
                  "objective (using `zs.variational.iw_objective` and then "
                  "calling its vimco() method.", category=FutureWarning)
    if axis is None:
        raise ValueError("vimco is a multi-sample objective, "
                         "the 'axis' argument must be specified.")

    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    joint_obs = merge_dicts(observed, latent_outputs)
    log_joint_value = log_joint(joint_obs)
    entropy = -sum(latent_logpdfs)
    l_signal = log_joint_value + entropy

    # check size along the sample axis
    err_msg = "vimco() is a multi-sample objective, " \
              "size along 'axis' in the objective should be larger than 1."
    if l_signal.get_shape()[axis:axis + 1].is_fully_defined():
        if l_signal.get_shape()[axis].value < 2:
            raise ValueError(err_msg)
    _assert_size_along_axis = tf.assert_greater_equal(
        tf.shape(l_signal)[axis], 2, message=err_msg)
    with tf.control_dependencies([_assert_size_along_axis]):
        l_signal = tf.identity(l_signal)

    # compute variance reduction term
    mean_except_signal = (tf.reduce_sum(l_signal, axis, keep_dims=True) -
                          l_signal) / tf.to_float(tf.shape(l_signal)[axis] - 1)
    x, sub_x = tf.to_float(l_signal), tf.to_float(mean_except_signal)

    n_dim = tf.rank(x)
    axis_dim_mask = tf.cast(tf.one_hot(axis, n_dim), tf.bool)
    original_mask = tf.cast(tf.one_hot(n_dim - 1, n_dim), tf.bool)
    axis_dim = tf.ones([n_dim], tf.int32) * axis
    originals = tf.ones([n_dim], tf.int32) * (n_dim - 1)
    perm = tf.where(original_mask, axis_dim, tf.range(n_dim))
    perm = tf.where(axis_dim_mask, originals, perm)
    multiples = tf.concat([tf.ones([n_dim], tf.int32), [tf.shape(x)[axis]]], 0)

    x = tf.transpose(x, perm=perm)
    sub_x = tf.transpose(sub_x, perm=perm)
    x_ex = tf.tile(tf.expand_dims(x, n_dim), multiples)
    x_ex = x_ex - tf.matrix_diag(x) + tf.matrix_diag(sub_x)
    control_variate = tf.transpose(log_mean_exp(x_ex, n_dim - 1), perm=perm)

    # variance reduced objective
    l_signal = log_mean_exp(l_signal, axis, keep_dims=True) - control_variate
    fake_term = tf.reduce_sum(-entropy * tf.stop_gradient(l_signal), axis)
    lower_bound = log_mean_exp(log_joint_value + entropy, axis)
    cost = -fake_term - log_mean_exp(log_joint_value + entropy, axis)

    return cost, lower_bound
