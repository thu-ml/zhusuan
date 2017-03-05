#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from .base import *


class Normal(UnivariateDistribution, ContinuousDistribution):
    """
    The class of univariate Normal distribution.

    :param mean: A Tensor. The mean of the Normal distribution. Should be
        broadcastable to match `logstd`.
    :param logstd: A Tensor. The log standard deviation of the Normal
        distribution. Should be broadcastable to match `mean`.
    :param event_n_dims: A 0-D `int32` Tensor representing number of
        dimensions of a single event. Default is 0, which means a univariate
        distribution. This is for common cases where we put a group of
        univariate random variables in a single event, so that their
        probabilities are calculated together. For example, when set to 1,
        the last dimension is treated as a single event, the log probabilities
        should sum over this axis.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 mean,
                 logstd,
                 event_ndims=0,
                 is_reparameterized=True):
        self.mean = tf.convert_to_tensor(mean)
        self.logstd = tf.convert_to_tensor(logstd)
        if not is_reparameterized:
            self.mean = tf.stop_gradient(self.mean)
            self.logstd = tf.stop_gradient(self.logstd)
        # TODO: check_numerics for logstd
        super(Normal, self).__init__(dtype=tf.float32,
                                     event_ndims=event_ndims,
                                     is_reparameterized=is_reparameterized)

    def _event_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_event_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.mean),
                                          tf.shape(self.logstd))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.mean.get_shape(),
                                         self.logstd.get_shape())

    def _mean_and_logstd(self):
        try:
            mean = tf.ones_like(self.logstd) * self.mean
            logstd = tf.ones_like(self.mean) * self.logstd
        except ValueError:
            raise ValueError("mean and logstd cannot be broadcast to have"
                             "the same shape.")
        mean = tf.expand_dims(mean, 0)
        logstd = tf.expand_dims(logstd, 0)
        return mean, logstd

    def _sample(self, n_samples):
        mean, logstd = self._mean_and_logstd()
        shape = tf.concat([[n_samples], self._batch_shape()], 0)
        return tf.random_normal(shape) * tf.exp(logstd) + mean

    def _log_prob(self, given):
        mean, logstd = self._mean_and_logstd()
        c = -0.5 * np.log(2 * np.pi)
        acc = tf.exp(-2 * logstd)
        return c - logstd - 0.5 * acc * tf.square(given - mean)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Bernoulli(UnivariateDistribution, DiscreteDistribution):
    """
    The class of univariate Bernoulli distribution.


    :param logits: A Tensor. The unnormalized log probabilities of being 1.

        .. math:: \\mathrm{logits}=\\log \\frac{p}{1 - p}

    :param event_n_dims: A 0-D `int32` Tensor representing number of
        dimensions of a single event. Default is 0, which means a univariate
        distribution. This is for common cases where we put a group of
        univariate random variables in a single event, so that their
        probabilities are calculated together. For example, when set to 1,
        the last dimension is treated as a single event, the log probabilities
        should sum over this axis.
    """
    def __init__(self,
                 logits,
                 event_n_dims=0):
        super(Bernoulli, self).__init__(dtype=tf.int32,
                                        event_ndims=event_n_dims)


class Categorical(UnivariateDistribution, DiscreteDistribution):
    """
    The class of univariate Categorical distribution.

    :param logits: A N-D (N >= 1) Tensor of shape (..., n_categories).
        Each slice `[i, j,..., k, :]` represents the un-normalized log
        probabilities for all categories.
    :param event_ndims: A 0-D `int32` Tensor representing number of
        dimensions of a single event. Default is 0, which means a univariate
        distribution. This is for common cases where we put a group of
        univariate random variables in a single event, so that their
        probabilities are calculated together. For example, when set to 1,
        the last dimension is treated as a single event, the log probabilities
        should sum over this axis.

    A single sample is a (N-1)-D Tensor with `tf.int32` values in range
    [0, n_categories).
    """
    def __init__(self,
                 name,
                 logits,
                 event_ndims=0):
        super(Categorical, self).__init__(dtype=tf.int32,
                                          event_ndims=event_ndims)


class Uniform(UnivariateDistribution, ContinuousDistribution):
    """
    The class of univariate Uniform distribution.

    :param minval: A Tensor. The lower bound on the range of the uniform
        distribution.
    :param maxval: A Tensor. The upper bound on the range of the uniform
        distribution. Should be element-wise  bigger than `minval`.
    :param event_ndims: A 0-D `int32` Tensor representing number of
        dimensions of a single event. Default is 0, which means a univariate
        distribution. This is for common cases where we put a group of
        univariate random variables in a single event, so that their
        probabilities are calculated together. For example, when set to 1,
        the last dimension is treated as a single event, the log probabilities
        should sum over this axis.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    """

    def __init__(self,
                 minval,
                 maxval,
                 event_ndims=0,
                 is_reparameterized=True):
        super(Uniform, self).__init__(dtype=tf.float32,
                                      event_ndims=event_ndims,
                                      is_reparameterized=is_reparameterized)
