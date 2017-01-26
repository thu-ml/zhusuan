#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import six
from six.moves import zip, map
from tensorflow.python.training import moving_averages

from .utils import log_mean_exp
from .evaluation import is_loglikelihood


def advi(model, observed, latent, reduction_indices=1, given=None):
    """
    Implements the automatic differentiation variational inference (ADVI)
    algorithm. For now we assume all latent variables have been transformed in
    the model definition to have their support on R^n.

    :param model: A model object that has a method logprob(latent, observed)
        to compute the log joint likelihood of the model.
    :param observed: A dictionary of (string, Tensor) pairs. Given inputs to
        the observed variables.
    :param latent: A dictionary of (string, (Tensor, Tensor)) pairs. The
        value of two Tensors represents (output, logpdf) given by the
        `zhusuan.layers.get_output` function for distribution layers.
    :param reduction_indices: The sample dimension(s) to reduce when
        computing the variational lower bound.
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
    lower_bound = model.log_prob(latent_outputs, observed, given) - \
        sum(latent_logpdfs)
    lower_bound = tf.reduce_mean(lower_bound, reduction_indices)
    return lower_bound


def iwae(model, observed, latent, reduction_indices=1, given=None):
    """
    Implements the importance weighted lower bound from (Burda, 2015).

    :param model: A model object that has a method logprob(latent, observed)
        to compute the log joint likelihood of the model.
    :param observed: A dictionary of (string, Tensor) pairs. Given inputs to
        the observed variables.
    :param latent: A dictionary of (string, (Tensor, Tensor)) pairs. The
        value of two Tensors represents (output, logpdf) given by the
        `zhusuan.layers.get_output` function for distribution layers.
    :param reduction_indices: The sample dimension(s) to reduce when
        computing the variational lower bound.
    :param given: A dictionary of (string, Tensor) pairs. This is used when
        some deterministic transformations have been computed in variational
        posterior and can be reused when evaluating model joint log likelihood.
        This dictionary will be directly passed to the model object.

    :return: A Tensor. The importance weighted lower bound.
    """
    return is_loglikelihood(model, observed, latent, reduction_indices, given)


def rws(model, observed, latent, reduction_indices=1, given=None):
    """
    Implements Reweighted Wake-sleep from (Bornschein, 2015).

    :param model: A model object that has a method logprob(latent, observed)
        to compute the log joint likelihood of the model.
    :param observed: A dictionary of (string, Tensor) pairs. Given inputs to
        the observed variables.
    :param latent: A dictionary of (string, (Tensor, Tensor)) pairs. The
        value of two Tensors represents (output, logpdf) given by the
        `zhusuan.layers.get_output` function for distribution layers.
    :param reduction_indices: The sample dimension(s) to reduce when
        computing the log likelihood.
    :param given: A dictionary of (string, Tensor) pairs. This is used when
        some deterministic transformations have been computed in the proposal
        and can be reused when evaluating model joint log likelihood.
        This dictionary will be directly passed to the model object.

    :return: A 1-D Tensor. The cost to minimize given by Reweighted Wake-sleep.
    :return: A 1-D Tensor of shape (batch_size,). Estimated log likelihoods.
    """
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    given = given if given is not None else {}
    log_joint = model.log_prob(latent_outputs, observed, given)
    entropy = -sum(latent_logpdfs)
    log_w = log_joint + entropy
    log_w_max = tf.reduce_max(log_w, reduction_indices, keep_dims=True)
    w_u = tf.exp(log_w - log_w_max)
    w_tilde = tf.stop_gradient(w_u / tf.reduce_sum(w_u, reduction_indices,
                                                   keep_dims=True))
    log_likelihood = log_mean_exp(log_w, reduction_indices)
    fake_log_joint_cost = -tf.reduce_sum(w_tilde*log_joint, reduction_indices)
    fake_proposal_cost = tf.reduce_sum(w_tilde*entropy, reduction_indices)
    cost = fake_log_joint_cost + fake_proposal_cost
    return cost, log_likelihood


def nvil(model, observed, latent, baseline=None, reduction_indices=1,
         given=None, variance_normalization=False, alpha=0.8):
    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    given = given if given is not None else {}
    log_joint = model.log_prob(latent_outputs, observed, given)  # log p(x,h)
    entropy = -sum(latent_logpdfs)  # -log q(h|x)
    l_signal = log_joint + entropy
    cost = 0.
    if baseline is not None:
        baseline = tf.expand_dims(baseline, reduction_indices)
        baseline_cost = 0.5 * tf.reduce_mean(tf.square(
            tf.stop_gradient(l_signal) - baseline), reduction_indices)
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

    fake_log_joint_cost = -tf.reduce_mean(log_joint, reduction_indices)
    fake_variational_cost = tf.reduce_mean(
        tf.stop_gradient(l_signal) * entropy, reduction_indices)
    cost += fake_log_joint_cost + fake_variational_cost
    lower_bound = tf.reduce_mean(log_joint + entropy, reduction_indices)
    return cost, lower_bound


def vimco(model, observed, latent, reduction_indices=1, given=None, is_particle_larger_one = False):
    if not is_particle_larger_one:
        return advi(model, observed, latent, reduction_indices, given)

    latent_k, latent_v = map(list, zip(*six.iteritems(latent)))
    latent_outputs = dict(zip(latent_k, map(lambda x: x[0], latent_v)))
    latent_logpdfs = map(lambda x: x[1], latent_v)
    given = given if given is not None else {}
    log_joint = model.log_prob(latent_outputs, observed, given)  # log p(x,h)
    entropy = -sum(latent_logpdfs)  # -log q(h|x)

    l_signal = log_joint + entropy
    mean_except_signal = (tf.reduce_sum(l_signal, reduction_indices, keep_dims=True) - l_signal) /\
                         tf.to_float(tf.shape(l_signal)[reduction_indices] - 1)

    # calculate log_mean_exp_sub
    x = tf.cast(l_signal, dtype=tf.float32)
    sub_x = tf.cast(mean_except_signal, dtype=tf.float32)
    x_shape = x.get_shape()
    n_dim = x_shape.ndims
    op_indices = n_dim - 1

    perm = []
    rep_para = []
    for i in range(n_dim):
        perm.append(i)
        rep_para.append(1)
    perm[n_dim - 1] = reduction_indices
    perm[reduction_indices] = op_indices

    x = tf.transpose(x, perm=perm)
    sub_x = tf.transpose(sub_x, perm=perm)
    K = tf.shape(x)[op_indices]
    rep_para.append(K)

    # extend to another dimension
    x_ex = tf.tile(tf.expand_dims(x, n_dim), rep_para)
    x_ex = x_ex - tf.matrix_diag(x) + tf.matrix_diag(sub_x)

    pre_signal = tf.transpose(log_mean_exp(x_ex, op_indices), perm = perm)
    # end of calculation of log_mean_exp_sub

    l_signal = log_mean_exp(l_signal, reduction_indices, keep_dims=True) - pre_signal

    fake_term = tf.reduce_sum(-entropy * \
                      tf.stop_gradient(l_signal), reduction_indices)
    lower_bound = log_mean_exp(log_joint + entropy, reduction_indices)
    object_function = fake_term + log_mean_exp(log_joint + entropy, reduction_indices)

    return object_function, lower_bound
