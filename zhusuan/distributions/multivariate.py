#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

from .base import *
from .utils import explicit_broadcast, log_combination, is_same_dynamic_shape


__all__ = [
    'Multinomial',
    'OnehotCategorical',
    'OnehotDiscrete',
    'Dirichlet',
]


class Multinomial(Distribution):
    """
    The class of Multinomial distribution.

    :param logits: A N-D (N >= 1) `float32` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param n_experiments: A 0-D `int32` Tensor. The number of experiments
        for each sample.
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`Distribution` for more detailed explanation.

    A single sample is a N-D Tensor with the same shape as logits. Each slice
    `[i, j, ..., k, :]` is a vector of counts for all categories.
    """

    def __init__(self, logits, n_experiments, group_event_ndims=0):
        self._logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        static_logits_shape = self._logits.get_shape()
        shape_err_msg = "logits should have rank >= 1."
        if static_logits_shape and (static_logits_shape.ndims < 1):
            raise ValueError(shape_err_msg)
        elif static_logits_shape and (
                static_logits_shape[-1].value is not None):
            self._n_categories = static_logits_shape[-1].value
        else:
            _assert_shape_op = tf.assert_rank_at_least(
                self._logits, 1, message=shape_err_msg)
            with tf.control_dependencies([_assert_shape_op]):
                self._logits = tf.identity(self._logits)
            self._n_categories = tf.shape(self._logits)[-1]

        sign_err_msg = "n_experiments must be positive"
        if isinstance(n_experiments, int):
            if n_experiments <= 0:
                raise ValueError(sign_err_msg)
            self._n_experiments = n_experiments
        else:
            n_experiments = tf.convert_to_tensor(n_experiments, tf.int32)
            _assert_rank_op = tf.assert_rank(
                n_experiments, 0,
                message="n_experiments should be a scalar (0-D Tensor).")
            _assert_positive_op = tf.assert_greater(
                n_experiments, 0, message=sign_err_msg)
            with tf.control_dependencies([_assert_rank_op,
                                          _assert_positive_op]):
                self._n_experiments = tf.identity(n_experiments)

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

    @property
    def n_experiments(self):
        """The number of experiments for each sample."""
        return self._n_experiments

    def _value_shape(self):
        return tf.convert_to_tensor([self.n_categories], tf.int32)

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
        if self.logits.get_shape().ndims == 2:
            logits_flat = self.logits
        else:
            logits_flat = tf.reshape(self.logits, [-1, self.n_categories])
        samples_flat = tf.transpose(
            tf.multinomial(logits_flat, n_samples * self.n_experiments))
        shape = tf.concat([[n_samples, self.n_experiments],
                           self.batch_shape], 0)
        samples = tf.reshape(samples_flat, shape)
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        static_n_exps = self.n_experiments if isinstance(self.n_experiments,
                                                         int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples, static_n_exps]).
                concatenate(self.get_batch_shape()))
        samples = tf.reduce_sum(
            tf.one_hot(samples, self.n_categories, dtype=tf.int32), axis=1)
        return samples

    def _log_prob(self, given):
        logits = self.logits
        if not (given.get_shape() and logits.get_shape()):
            given, logits = explicit_broadcast(given, logits,
                                               'given', 'logits')
        else:
            if given.get_shape().ndims != logits.get_shape().ndims:
                given, logits = explicit_broadcast(given, logits,
                                                   'given', 'logits')
            elif given.get_shape().is_fully_defined() and \
                    logits.get_shape().is_fully_defined():
                if given.get_shape() != logits.get_shape():
                    given, logits = explicit_broadcast(given, logits,
                                                       'given', 'logits')
            else:
                # Below code seems to induce a BUG when this function is
                # called in HMC. Probably due to tensorflow's not supporting
                # control flow edge from an op inside the body to outside.
                # We should further fix this.
                #
                # given, logits = tf.cond(
                #     is_same_dynamic_shape(given, logits),
                #     lambda: (given, logits),
                #     lambda: explicit_broadcast(given, logits,
                #                                'given', 'logits'))
                given, logits = explicit_broadcast(given, logits,
                                                   'given', 'logits')
        normalized_logits = logits - tf.reduce_logsumexp(
            logits, axis=-1, keep_dims=True)
        log_p = log_combination(self.n_experiments, given) + \
            tf.reduce_sum(tf.to_float(given) * normalized_logits, -1)
        return log_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


class OnehotCategorical(Distribution):
    """
    The class of one-hot Categorical distribution.

    :param logits: A N-D (N >= 1) `float32` Tensor of shape (...,
        n_categories). Each slice `[i, j, ..., k, :]` represents the
        un-normalized log probabilities for all categories.

        .. math:: \\mathrm{logits} \\propto \\log p

    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
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
        elif static_logits_shape and (
                static_logits_shape[-1].value is not None):
            self._n_categories = static_logits_shape[-1].value
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
        return tf.convert_to_tensor([self.n_categories], tf.int32)

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
        if self.logits.get_shape().ndims == 2:
            logits_flat = self.logits
        else:
            logits_flat = tf.reshape(self.logits, [-1, self.n_categories])
        samples_flat = tf.transpose(tf.multinomial(logits_flat, n_samples))
        if self.logits.get_shape().ndims == 2:
            samples = samples_flat
        else:
            shape = tf.concat([[n_samples], self.batch_shape], 0)
            samples = tf.reshape(samples_flat, shape)
            static_n_samples = n_samples if isinstance(n_samples,
                                                       int) else None
            samples.set_shape(
                tf.TensorShape([static_n_samples]).
                    concatenate(self.get_batch_shape()))
        samples = tf.one_hot(samples, self.n_categories, dtype=tf.int32)
        return samples

    def _log_prob(self, given):
        given = tf.to_float(given)
        logits = self.logits
        if not (given.get_shape() and logits.get_shape()):
            given, logits = explicit_broadcast(given, logits,
                                               'given', 'logits')
        else:
            if given.get_shape().ndims != logits.get_shape().ndims:
                given, logits = explicit_broadcast(given, logits,
                                                   'given', 'logits')
            elif given.get_shape().is_fully_defined() and \
                    logits.get_shape().is_fully_defined():
                if given.get_shape() != logits.get_shape():
                    given, logits = explicit_broadcast(given, logits,
                                                       'given', 'logits')
            else:
                # Below code seems to induce a BUG when this function is
                # called in HMC. Probably due to tensorflow's not supporting
                # control flow edge from an op inside the body to outside.
                # We should further fix this.
                #
                # given, logits = tf.cond(
                #     is_same_dynamic_shape(given, logits),
                #     lambda: (given, logits),
                #     lambda: explicit_broadcast(given, logits,
                #                                'given', 'logits'))
                given, logits = explicit_broadcast(given, logits,
                                                   'given', 'logits')
        if (given.get_shape().ndims == 2) or (logits.get_shape().ndims == 2):
            given_flat = given
            logits_flat = logits
        else:
            given_flat = tf.reshape(given, [-1, self.n_categories])
            logits_flat = tf.reshape(logits, [-1, self.n_categories])
        log_p_flat = -tf.nn.softmax_cross_entropy_with_logits(
            labels=given_flat, logits=logits_flat)
        if (given.get_shape().ndims == 2) or (logits.get_shape().ndims == 2):
            log_p = log_p_flat
        else:
            log_p = tf.reshape(log_p_flat, tf.shape(logits)[:-1])
            if given.get_shape() and logits.get_shape():
                log_p.set_shape(tf.broadcast_static_shape(
                    given.get_shape(), logits.get_shape())[:-1])
        return log_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))


OnehotDiscrete = OnehotCategorical


class Dirichlet(Distribution):
    """
    The class of Dirichlet distribution.

    :param alpha: A N-D (N >= 1) `float32` Tensor of shape (..., n_categories).
        Each slice `[i, j, ..., k, :]` represents the concentration parameter
        of a Dirichlet distribution. Should be positive.
    :param group_event_ndims: A 0-D `int32` Tensor representing the number of
        dimensions in `batch_shape` (counted from the end) that are grouped
        into a single event, so that their probabilities are calculated
        together. Default is 0, which means a single value is an event.
        See :class:`Distribution` for more detailed explanation.

    A single sample is a N-D Tensor with the same shape as alpha. Each slice
    `[i, j, ..., k, :]` of the sample is a vector of probabilities of a
    Categorical distribution `[x_1, x_2, ... ]`, which lies on the simplex

    .. math:: \\sum_{i} x_i = 1, 0 < x_i < 1

    """

    def __init__(self, alpha, group_event_ndims=0, check_numerics=False):
        self._alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        static_alpha_shape = self._alpha.get_shape()
        shape_err_msg = "alpha should have rank >= 1."
        cat_err_msg = "n_categories (length of the last axis " \
                      "of alpha) should be at least 2."
        if static_alpha_shape and (static_alpha_shape.ndims < 1):
            raise ValueError(shape_err_msg)
        elif static_alpha_shape and (
                static_alpha_shape[-1].value is not None):
            self._n_categories = static_alpha_shape[-1].value
            if self._n_categories < 2:
                raise ValueError(cat_err_msg)
        else:
            _assert_shape_op = tf.assert_rank_at_least(
                self._alpha, 1, message=shape_err_msg)
            with tf.control_dependencies([_assert_shape_op]):
                self._alpha = tf.identity(self._alpha)
            self._n_categories = tf.shape(self._alpha)[-1]

            _assert_cat_op = tf.assert_greater_equal(
                self._n_categories, 2, message=cat_err_msg)
            with tf.control_dependencies([_assert_cat_op]):
                self._alpha = tf.identity(self._alpha)
        self._check_numerics = check_numerics

        super(Dirichlet, self).__init__(
            dtype=tf.float32,
            is_continuous=True,
            is_reparameterized=False,
            group_event_ndims=group_event_ndims)

    @property
    def alpha(self):
        """The concentration parameter of the Dirichlet distribution."""
        return self._alpha

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._n_categories

    def _value_shape(self):
        return tf.convert_to_tensor([self.n_categories], tf.int32)

    def _get_value_shape(self):
        if isinstance(self.n_categories, int):
            return tf.TensorShape([self.n_categories])
        return tf.TensorShape([None])

    def _batch_shape(self):
        return tf.shape(self.alpha)[:-1]

    def _get_batch_shape(self):
        if self.alpha.get_shape():
            return self.alpha.get_shape()[:-1]
        return tf.TensorShape(None)

    def _sample(self, n_samples):
        samples = tf.random_gamma([n_samples], self.alpha, beta=1)
        return samples / tf.reduce_sum(samples, -1, keep_dims=True)

    def _log_prob(self, given):
        alpha = self.alpha
        if not (given.get_shape() and alpha.get_shape()):
            given, alpha = explicit_broadcast(given, alpha,
                                              'given', 'alpha')
        else:
            if given.get_shape().ndims != alpha.get_shape().ndims:
                given, alpha = explicit_broadcast(given, alpha,
                                                  'given', 'alpha')
            elif given.get_shape().is_fully_defined() and \
                    alpha.get_shape().is_fully_defined():
                if given.get_shape() != alpha.get_shape():
                    given, alpha = explicit_broadcast(given, alpha,
                                                      'given', 'alpha')
            else:
                # Below code seems to induce a BUG when this function is
                # called in HMC. Probably due to tensorflow's not supporting
                # control flow edge from an op inside the body to outside.
                # We should further fix this.
                #
                # given, alpha = tf.cond(
                #     is_same_dynamic_shape(given, alpha),
                #     lambda: (given, alpha),
                #     lambda: explicit_broadcast(given, alpha,
                #                                'given', 'alpha'))
                given, alpha = explicit_broadcast(given, alpha,
                                                  'given', 'alpha')
        log_Beta_alpha = tf.lbeta(alpha)
        # fix of no static shape inference for tf.lbeta
        if alpha.get_shape():
            log_Beta_alpha.set_shape(alpha.get_shape()[:-1])
        log_given = tf.log(given)
        if self._check_numerics:
            with tf.control_dependencies(
                    [tf.check_numerics(log_Beta_alpha, "lbeta(alpha)"),
                     tf.check_numerics(log_given, "log(given)")]):
                log_given = tf.identity(log_given)
        log_p = -log_Beta_alpha + tf.reduce_sum((alpha - 1) * log_given, -1)
        return log_p

    def _prob(self, given):
        return tf.exp(self._log_prob(given))
