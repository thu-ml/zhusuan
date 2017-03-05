#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf

from .base import *


__all__ = [
    'Normal',
    'Bernoulli',
    'Categorical',
    'Uniform',
    'Discrete',
]


class Normal(Distribution):
    """
    The class of univariate Normal distribution.

    :param mean: A Tensor. The mean of the Normal distribution. Should be
        broadcastable to match `logstd`.
    :param logstd: A Tensor. The log standard deviation of the Normal
        distribution. Should be broadcastable to match `mean`.
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is a event.
        See :class:`Distribution` for more detailed explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    :param check_numerics: Bool. Whether to check numeric issues.
    """

    def __init__(self,
                 mean,
                 logstd,
                 group_event_ndims=0,
                 is_reparameterized=True,
                 check_numerics=True):
        self._mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self._logstd = tf.convert_to_tensor(logstd, dtype=tf.float32)
        self._check_numerics = check_numerics
        super(Normal, self).__init__(
            dtype=tf.float32,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            group_event_ndims=group_event_ndims)

    @property
    def mean(self):
        """The mean of the Normal distribution."""
        return self._mean

    @property
    def logstd(self):
        """The log standard deviation of the Normal distribution."""
        return self._logstd

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.mean),
                                          tf.shape(self.logstd))

    def _get_batch_shape(self):
        return tf.broadcast_static_shape(self.mean.get_shape(),
                                         self.logstd.get_shape())

    def _sample(self, n_samples):
        mean = tf.expand_dims(self.mean, 0)
        logstd = tf.expand_dims(self.logstd, 0)
        if self.is_reparameterized:
            mean = tf.stop_gradient(mean)
            logstd = tf.stop_gradient(logstd)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        return tf.random_normal(shape) * tf.exp(logstd) + mean

    def _log_prob(self, given):
        mean = tf.expand_dims(self.mean, 0)
        logstd = tf.expand_dims(self.logstd, 0)
        c = -0.5 * np.log(2 * np.pi)
        precision = tf.exp(-2 * logstd)
        if self._check_numerics:
            with tf.control_dependencies(
                    [tf.check_numerics(precision, "precision")]):
                precision = tf.identity(precision)
        return c - logstd - 0.5 * precision * tf.square(given - mean)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Bernoulli(Distribution):
    """
    The class of univariate Bernoulli distribution.

    :param logits: A Tensor. The log-odds of probabilities of being 1.

        .. math:: \\mathrm{logits} = \\log \\frac{p}{1 - p}

    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is a event.
        See :class:`Distribution` for more detailed explanation.
    """
    def __init__(self, logits, group_event_ndims=0):
        self._logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        super(Bernoulli, self).__init__(
            dtype=tf.int32,
            is_continuous=False,
            is_reparameterized=False,
            group_event_ndims=group_event_ndims)

    @property
    def logits(self):
        """The log-odds of probabilities of being 1."""
        return self._logits

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.shape(self.logits)

    def _get_batch_shape(self):
        return self.logits.get_shape()

    def _sample(self, n_samples):
        logits = tf.expand_dims(self.logits, 0)
        p = tf.sigmoid(logits)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        alpha = tf.random_uniform(shape, minval=0, maxval=1)
        samples = tf.cast(tf.less(alpha, p), dtype=tf.float32)
        return samples

    def _log_prob(self, given):
        logits = tf.expand_dims(self.logits, 0)
        # TODO: check static/dynamic shape to avoid broadcasting
        try:
            logits = tf.ones_like(given) * logits
            given = tf.ones_like(logits) * given
        except ValueError:
            raise ValueError("given and logits cannot broadcast to have"
                             "the same shape. ({} vs. {})"
                             .format(given.get_shape(), logits.get_shape()))
        return -tf.nn.sigmoid_cross_entropy_with_logits(labels=given,
                                                        logits=logits)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class Categorical(Distribution):
    """
    The class of univariate Categorical distribution.

    :param logits: A N-D (N >= 1) Tensor of shape (..., n_categories).
        Each slice `[i, j,..., k, :]` represents the un-normalized log
        probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log \\frac{p}

    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is a event.
        See :class:`Distribution` for more detailed explanation.

    A single sample is a (N-1)-D Tensor with `tf.int32` values in range
    [0, n_categories).
    """
    def __init__(self, logits, group_event_ndims=0):
        self._logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        super(Categorical, self).__init__(
            dtype=tf.int32,
            is_continuous=False,
            is_reparameterized=False,
            group_event_ndims=group_event_ndims)

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.shape(self.logits)[:-1]

    def _get_batch_shape(self):
        static_logits_shape = self.logits.get_shape()
        if static_logits_shape is not None:
            return static_logits_shape[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        pass

    def _log_prob(self, given):
        pass

    def _prob(self, given):
        pass


Discrete = Categorical


class Uniform(Distribution):
    """
    The class of univariate Uniform distribution.

    :param minval: A Tensor. The lower bound on the range of the uniform
        distribution.
    :param maxval: A Tensor. The upper bound on the range of the uniform
        distribution. Should be element-wise  bigger than `minval`.
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is a event.
        See :class:`Distribution` for more detailed explanation.
    :param is_reparameterized: A Bool. If True, gradients on samples from this
        distribution are allowed to propagate into inputs, using the
        reparametrization trick from (Kingma, 2013).
    """

    def __init__(self,
                 minval,
                 maxval,
                 group_event_ndims=0,
                 is_reparameterized=True):
        super(Uniform, self).__init__(
            dtype=tf.float32,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            group_event_ndims=group_event_ndims)
