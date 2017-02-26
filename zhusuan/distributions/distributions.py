#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np

from zhusuan.utils import convert_to_int, add_name_scope


__all__ = [
    'Distribution',
]


class Distribution(object):
    """
    The :class:`Distribution` class is the base class for various probabilistic
    distributions which support batch inputs, generating batches of samples and
    evaluate probabilities at batches of given values.

    The typical input shapes for a `Distribution` is like
    `batch_shape + event_shape`. We borrow this terminology from
    `tf.contrib.distributions`, where `event_shape` represents the shape
    of non-batch input parameter, `batch_shape` represents how many independent
    inputs are fed into the distribution.

    Samples generated are of shape `[n + ]batch_shape + event_shape`. For
    `n=1`, the first additional axis is omitted.

    When evaluating probabilities at given values, the given Tensor can be of
    shape `[n + ]batch_shape + event_shape`. The returned Tensor has shape
    `[n + ]batch_shape`.

    :param dtype: The value type of samples from the distribution.
    :param is_continuous: Whether the distribution is continuous.
    """

    def __init__(self, dtype, is_continuous, *args, **kwargs):
        self._dtype = dtype
        self._is_continuous = is_continuous

    @property
    def dtype(self):
        """The sample type of the distribution."""
        return self._dtype

    @property
    def is_continuous(self):
        """Whether the distribution is continuous."""
        return self._is_continuous

    @property
    def event_shape(self):
        """
        The shape of a single sample from the distribution with non-batch
        input. For batch inputs, the shape of a generated sample is
        `batch_shape + event_shape`.
        We borrow this concept from `tf.contrib.distributions`.
        """
        return self._event_shape()

    def _event_shape(self):
        """
        Private method for subclasses to rewrite the `event_shape` property.
        """
        raise NotImplementedError()

    @property
    def batch_shape(self):
        """
        The shape showing how many independent inputs (which we call batches)
        are fed into the distribution. For batch inputs, the shape of a
        generated sample is `batch_shape + event_shape`.
        We borrow this concept from `tf.contrib.distributions`.
        """
        return self._batch_shape()

    def _batch_shape(self):
        """
        Private method for subclasses to rewrite the `batch_shape` property.
        """
        raise NotImplementedError()

    def sample(self, n_samples=1):
        """
        Return samples from the distribution. For `n_samples` larger than 1,
        the returned Tensor has a new sample dimension with size `n_samples`
        inserted at `axis=0`.

        :param n_samples: A 0-D `int32` Tensor. How many independent samples
            to draw from the distribution.
        :return: A Tensor of samples.
        """
        static_n_samples = convert_to_int(n_samples)
        n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
        samples = self._sample(n_samples)
        if static_n_samples is not None:
            if static_n_samples == 1:
                return tf.reshape(samples, tf.concat([self.batch_shape,
                                                      self.event_shape], 0))
            return samples
        else:
            return tf.cond(
                n_samples > 1,
                lambda: tf.reshape(samples, tf.concat([self.batch_shape,
                                                       self.event_shape], 0)),
                lambda: samples)

    def log_prob(self, given):
        """
        Compute log probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate log probability
            density (mass) function. Must be able to broadcast to have a shape
            of `[n + ]batch_shape + event_shape`.
        :return: A Tensor of shape `[n + ]batch_shape`.
        """


    def prob(self, given):
        """
        Compute probability density (mass) function at `given` value.

        :param given: A Tensor. The value at which to evaluate probability
            density (mass) function.
        :return: A Tensor.
        """

    def _sample(self, n_samples):
        """
        Private method for subclasses to rewrite the `sample` method.
        """
        raise NotImplementedError()

    def _log_prob(self, given):
        """
        Private method for subclasses to rewrite the `log_prob` method.
        """
        raise NotImplementedError()

    def _prob(self, given):
        """
        Private method for subclasses to rewrite the `prob` method.
        """
        raise NotImplementedError()


class ContinuousDistribution(Distribution):
    """
    Base class of continuous distributions.

    :param dtype: The value type of samples from the distribution.
    :param is_reparameterized: A bool. Whether the gradients of samples can
        and are allowed to propagate back into inputs, using the
        reparametrization trick from (Kingma, 2013).
    """

    def __init__(self, dtype, is_reparameterized, *args, **kwargs):
        self._is_reparameterized = is_reparameterized
        super(ContinuousDistribution, self).__init__(
            dtype, is_continuous=True, *args, **kwargs)

    @property
    def is_reparameterized(self):
        """
        Whether the gradients of samples can and are allowed to propagate back
        into inputs, using the reparametrization trick from (Kingma, 2013).
        """
        return self._is_reparameterized


class DiscreteDistribution(Distribution):
    """
    Base class of discrete distributions.

    :param dtype: The value type of samples from the distribution.
    """

    def __init__(self, dtype, *args, **kwargs):
        super(DiscreteDistribution, self).__init__(
            dtype, is_continuous=False, *args, **kwargs)
