#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .base import *


__all__ = [
    'Multinomial',
    'OnehotCategorical',
    'OnehotDiscrete',
]


class Multinomial(Distribution):
    """
    The class of Multinomial distribution.

    :param logits: A N-D (N >= 1) Tensor of shape (..., n_categories).
        Each slice `[i, j, ..., k, :]` represents the un-normalized log
        probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log \\frac{p}

    :param n: A Tensor that can be broadcast to match `logits[:-1]`. The
        number of experiments for each sample.
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is a event.
        See :class:`Distribution` for more detailed explanation.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a vector of counts for all categories.
    """

    def __init__(self, logits, group_event_ndims=0):
        self._logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        static_logits_shape = self._logits.get_shape()
        shape_err_msg = "logits should have rank >= 1."
        if static_logits_shape and (static_logits_shape.ndims < 1):
            raise ValueError(shape_err_msg)
        elif static_logits_shape and (static_logits_shape[-1]):
            self._n_categories = static_logits_shape[-1]
        else:
            _assert_shape_op = tf.assert_rank_at_least(
                self._logits, 1, message=shape_err_msg)
            with tf.control_dependencies([_assert_shape_op]):
                self._logits = tf.identity(self._logits)
            self._n_categories = tf.shape(self._logits)[-1]

        super(Multinomial, self).__init__(
            dtype=tf.int32,
            is_continuous=False,
            is_reparameterized=False,
            group_event_ndims=group_event_ndims)

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _value_shape(self):
        return tf.convert_to_tensor(self.n_categories, tf.int32)

    def _get_value_shape(self):
        if isinstance(self.n_categories, int):
            return tf.TensorShape([self.n_categories])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.logits)[:-1]

    def _get_batch_shape(self):
        if self.logits.get_shape():
            return self.logits.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        logits_flat = tf.reshape(self.logits, [-1, self.n_categories])
        samples_flat = tf.multinomial(logits_flat, n_samples)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        samples = tf.reshape(tf.transpose(samples_flat), shape)
        return samples

    def _log_prob(self, given):
        logits = self.logits

        def _broadcast(given, logits):
            try:
                given *= tf.ones_like(logits[:-1])
                logits *= tf.ones_like(tf.expand_dims(given, -1))
            except ValueError:
                raise ValueError(
                    "given and logits[:-1] cannot broadcast to match. ("
                    "{} vs. {}[:-1])".format(given.get_shape(),
                                             logits.get_shape()))

        if given.get_shape().is_fully_defined() and \
                logits.get_shape().is_fully_defined():
            if given.get_shape() != self.logits.get_shape()[:-1]:
                given, logits = _broadcast(given, logits)
        else:
            given, logits = tf.cond(
                tf.equal(tf.shape(given), tf.shape(logits)[:-1]),
                lambda: (given, logits),
                lambda: _broadcast(given, logits))
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=given,
                                                               logits=logits)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class OnehotCategorical(Distribution):
    """
    The class of one-hot Categorical distribution.

    :param logits: A N-D (N >= 1) Tensor of shape (..., n_categories).
        Each slice `[i, j, ..., k, :]` represents the un-normalized log
        probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log \\frac{p}

    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is a event.
        See :class:`Distribution` for more detailed explanation.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a one-hot vector of the selected category.
    """

    def __init__(self, logits, group_event_ndims=0):
        self._logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        static_logits_shape = self._logits.get_shape()
        shape_err_msg = "logits should have rank >= 1."
        if static_logits_shape and (static_logits_shape.ndims < 1):
            raise ValueError(shape_err_msg)
        elif static_logits_shape and (static_logits_shape[-1]):
            self._n_categories = static_logits_shape[-1]
        else:
            _assert_shape_op = tf.assert_rank_at_least(
                self._logits, 1, message=shape_err_msg)
            with tf.control_dependencies([_assert_shape_op]):
                self._logits = tf.identity(self._logits)
            self._n_categories = tf.shape(self._logits)[-1]

        super(OnehotCategorical, self).__init__(
            dtype=tf.int32,
            is_continuous=False,
            is_reparameterized=False,
            group_event_ndims=group_event_ndims)

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _value_shape(self):
        return tf.constant([], dtype=tf.int32)

    def _get_value_shape(self):
        return tf.TensorShape([])

    def _batch_shape(self):
        return tf.shape(self.logits)[:-1]

    def _get_batch_shape(self):
        if self.logits.get_shape():
            return self.logits.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        logits_flat = tf.reshape(self.logits, [-1, self.n_categories])
        samples_flat = tf.multinomial(logits_flat, n_samples)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        samples = tf.reshape(tf.transpose(samples_flat), shape)
        return samples

    def _log_prob(self, given):
        logits = self.logits

        def _broadcast(given, logits):
            try:
                given *= tf.ones_like(logits[:-1])
                logits *= tf.ones_like(tf.expand_dims(given, -1))
            except ValueError:
                raise ValueError(
                    "given and logits[:-1] cannot broadcast to match. ("
                    "{} vs. {}[:-1])".format(given.get_shape(),
                                             logits.get_shape()))

        if given.get_shape().is_fully_defined() and \
                logits.get_shape().is_fully_defined():
            if given.get_shape() != self.logits.get_shape()[:-1]:
                given, logits = _broadcast(given, logits)
        else:
            given, logits = tf.cond(
                tf.equal(tf.shape(given), tf.shape(logits)[:-1]),
                lambda: (given, logits),
                lambda: _broadcast(given, logits))
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=given,
                                                               logits=logits)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


OnehotDiscrete = OnehotCategorical
