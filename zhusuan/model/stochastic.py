#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import distributions

from .base import StochasticTensor


__all__ = [
    'Uniform',
    'Normal',
    'Bernoulli',
    'Categorical',
    'Discrete',
    'OnehotCategorical',
    'OnehotDiscrete',
]


class OnehotCategorical(MultivariateStochasticTensor,
                        ContinuousStochasticTensor):
    """
    The class of one hot Categorical StochasticTensor.

    :param logits: A N-D (N >= 1) Tensor of shape (..., n_categories).
        Each slice `[i, j,..., k, :]` represents the un-normalized log
        probabilities for all categories.
    :param sample_shape: A Tensor. The shape of sample dimensions, which
        indicates how many independent samples to generate.
    :param validate_args: A Bool. Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.

    A single sample is a N-D one-hot Tensor of shape (..., n_categories).
    """
    def __init__(self, name, dtype, is_reparameterized, *args):
        with tf.control_dependencies([tf.assert_rank_at_least(logits, 1)]):
            self._logits = tf.identity(logits)
        self._categorical = distributions.Categorical(
            logits=self._logits,
            validate_args=validate_args,
            allow_nan_stats=False)
        super(OnehotCategorical, self).__init__(name=name, dtype=tf.int32,
                                                is_reparameterized=event_axis,
                                                event_axis=event_axis,
                                                sample_shape=sample_shape)

    def _sample(self):
        return tf.one_hot(self._categorical.sample(self.sample_shape),
                          tf.shape(self._logits)[-1])

    def _log_prob(self, given):
        pass

    def _prob(self):
        pass


# alias
Discrete = Categorical
OnehotDiscrete = OnehotCategorical
